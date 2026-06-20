"""
音频采集与FFT处理引擎 - Android修复版
使用pyjnius调用Android原生AudioRecord API
"""
import numpy as np
import threading
import queue
from collections import deque


class AudioSpectrumEngine:
    """
    音频频谱分析引擎
    负责麦克风音频采集、FFT计算、峰值检测
    """

    @staticmethod
    def _find_peaks_numpy(magnitude, height=None, distance=None):
        """用 numpy 实现的简化版 find_peaks"""
        if len(magnitude) < 3:
            return np.array([])

        left = magnitude[1:-1] > magnitude[:-2]
        right = magnitude[1:-1] > magnitude[2:]
        local_maxima = np.where(left & right)[0] + 1

        if len(local_maxima) == 0:
            return np.array([])

        if height is not None:
            peaks = local_maxima[magnitude[local_maxima] > height]
        else:
            peaks = local_maxima

        if len(peaks) == 0:
            return np.array([])

        if distance is not None and distance >= 1:
            sorted_idx = np.argsort(magnitude[peaks])[::-1]
            keep = []
            for p in peaks[sorted_idx]:
                if len(keep) == 0:
                    keep.append(p)
                else:
                    min_dist = min(abs(p - k) for k in keep)
                    if min_dist >= distance:
                        keep.append(p)
            peaks = np.array(sorted(keep))

        return peaks

    def __init__(self, samplerate=48000, blocksize=2048, channels=1):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels

        self.is_running = False
        self.is_paused = False

        self.audio_queue = queue.Queue(maxsize=20)
        self._audio_buffer = deque(maxlen=samplerate * 2)

        self.fft_size = 2048
        self.hop_size = 1024
        self.window_type = 'hann'
        self.dc_remove = True
        self.use_welch = False

        self.freq_min = 0
        self.freq_max = 20000

        self.amp_max = 1.0
        self.log_scale = False

        self.peak_count = 3

        self.spectrum_avg = False
        self._spectrum_buffer = deque(maxlen=10)

        self._window_cache = {}

        self.latest_freqs = None
        self.latest_magnitude = None
        self.latest_time_data = None
        self.lock = threading.Lock()

        self._capture_thread = None
        self._stream = None

        # Android AudioRecord 相关
        self._audio_record = None
        self._record_thread = None
        self._is_android = False
        try:
            from jnius import autoclass
            self._is_android = True
        except ImportError:
            pass

    def _get_window(self, size, window_type):
        key = (size, window_type)
        if key not in self._window_cache:
            if window_type == 'hann':
                self._window_cache[key] = np.hanning(size)
            elif window_type == 'hamming':
                self._window_cache[key] = np.hamming(size)
            elif window_type == 'blackman':
                self._window_cache[key] = np.blackman(size)
            else:
                self._window_cache[key] = np.ones(size)
        return self._window_cache[key]

    def _compute_single_fft(self, data):
        if self.dc_remove:
            data = data - np.mean(data)
        fft_size = min(self.fft_size, len(data))
        window = self._get_window(fft_size, self.window_type)
        if len(data) >= fft_size:
            segment = data[:fft_size] * window
        else:
            segment = np.zeros(fft_size)
            segment[:len(data)] = data * window[:len(data)]
        fft_result = np.fft.rfft(segment)
        magnitude = np.abs(fft_result) / fft_size * 2
        if len(magnitude) > 0:
            magnitude[0] /= 2
        freqs = np.fft.rfftfreq(fft_size, 1 / self.samplerate)
        return freqs, magnitude

    def _welch_average(self, data):
        if self.dc_remove:
            data = data - np.mean(data)

        fft_size = self.fft_size
        hop_size = self.hop_size
        window = self._get_window(fft_size, self.window_type)

        n_frames = max(1, (len(data) - fft_size) // hop_size + 1)
        spectrum_sum = None

        for i in range(n_frames):
            start = i * hop_size
            end = start + fft_size
            if end > len(data):
                break
            frame = data[start:end] * window
            fft_result = np.fft.rfft(frame)
            mag = np.abs(fft_result) / fft_size * 2
            if len(mag) > 0:
                mag[0] /= 2
            if spectrum_sum is None:
                spectrum_sum = mag
                freqs = np.fft.rfftfreq(fft_size, 1 / self.samplerate)
            else:
                spectrum_sum += mag

        if spectrum_sum is not None and n_frames > 0:
            magnitude = spectrum_sum / n_frames
        else:
            freqs, magnitude = self._compute_single_fft(data)

        return freqs, magnitude

    def process_audio(self, audio_data):
        """处理一帧音频数据，计算FFT"""
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.frombuffer(audio_data, dtype=np.float32)

        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]

        self._audio_buffer.extend(audio_data)

        buffer_data = np.array(list(self._audio_buffer)[-self.fft_size:])
        if len(buffer_data) < 64:
            return None, None

        if self.use_welch and len(buffer_data) > self.fft_size:
            freqs, magnitude = self._welch_average(buffer_data)
        else:
            freqs, magnitude = self._compute_single_fft(buffer_data)

        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max) & (freqs >= 0)
        freqs = freqs[mask]
        magnitude = magnitude[mask]

        if self.spectrum_avg:
            if len(self._spectrum_buffer) > 0 and len(self._spectrum_buffer[0]) != len(magnitude):
                self._spectrum_buffer.clear()
            self._spectrum_buffer.append(magnitude)
            if len(self._spectrum_buffer) > 1:
                magnitude = np.mean(list(self._spectrum_buffer), axis=0)

        if self.log_scale:
            magnitude = 20 * np.log10(magnitude + 1e-10)

        with self.lock:
            self.latest_freqs = freqs
            self.latest_magnitude = magnitude
            self.latest_time_data = buffer_data[-self.blocksize:] if len(buffer_data) >= self.blocksize else buffer_data

        return freqs, magnitude

    def find_peaks(self, freqs=None, magnitude=None):
        """检测频谱峰值"""
        if freqs is None:
            with self.lock:
                freqs = self.latest_freqs
                magnitude = self.latest_magnitude

        if freqs is None or magnitude is None or len(magnitude) == 0:
            return []

        if self.peak_count <= 0:
            return []

        if self.log_scale:
            threshold = np.max(magnitude) - 20
        else:
            threshold = np.max(magnitude) * 0.25

        peaks = self._find_peaks_numpy(magnitude, height=threshold, distance=max(1, len(freqs)//50))
        if len(peaks) == 0:
            return []

        top_indices = peaks[np.argsort(magnitude[peaks])[-self.peak_count:]]
        return [(freqs[idx], magnitude[idx]) for idx in top_indices]

    def start_capture(self, stream_callback=None):
        """开始音频采集"""
        self.is_running = True
        self.is_paused = False
        self._start_audio_stream(stream_callback)

    def _start_audio_stream(self, stream_callback=None):
        """启动音频流 - Android使用AudioRecord，桌面使用pyaudio"""
        if self._is_android:
            self._start_android_audio(stream_callback)
        else:
            self._start_pyaudio_stream(stream_callback)

    def _start_android_audio(self, stream_callback=None):
        """使用pyjnius调用Android原生AudioRecord"""
        try:
            from jnius import autoclass, JavaException

            AudioRecord = autoclass('android.media.AudioRecord')
            AudioSource = autoclass('android.media.MediaRecorder$AudioSource')
            AudioFormat = autoclass('android.media.AudioFormat')

            # 计算最小缓冲区大小
            min_buffer_size = AudioRecord.getMinBufferSize(
                self.samplerate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            if min_buffer_size < 0:
                print(f"[音频引擎] AudioRecord.getMinBufferSize 失败: {min_buffer_size}")
                self.is_running = False
                return

            # 使用更大的缓冲区避免欠载
            buffer_size = max(min_buffer_size, self.blocksize * 2)

            # 创建AudioRecord实例
            self._audio_record = AudioRecord(
                AudioSource.MIC,
                self.samplerate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                buffer_size
            )

            # 检查初始化状态
            if self._audio_record.getState() != AudioRecord.STATE_INITIALIZED:
                print("[音频引擎] AudioRecord 初始化失败，请检查麦克风权限")
                self._audio_record.release()
                self._audio_record = None
                self.is_running = False
                return

            # 开始录音
            self._audio_record.startRecording()
            print(f"[音频引擎] Android AudioRecord 已启动，采样率: {self.samplerate}")

            # 启动采集线程
            self._record_thread = threading.Thread(
                target=self._android_record_loop,
                args=(buffer_size, stream_callback),
                daemon=True
            )
            self._record_thread.start()

        except Exception as e:
            print(f"[音频引擎] Android音频启动失败: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False

    def _android_record_loop(self, buffer_size, stream_callback=None):
        """Android音频采集线程"""
        try:
            from jnius import JavaException

            # 创建字节缓冲区
            audio_buffer = bytearray(buffer_size)

            while self.is_running and not self.is_paused:
                try:
                    # 读取音频数据
                    bytes_read = self._audio_record.read(audio_buffer, 0, buffer_size)

                    if bytes_read > 0:
                        # 转换为numpy数组 (16位PCM)
                        audio_data = np.frombuffer(bytes(audio_buffer[:bytes_read]), dtype=np.int16)
                        audio_data = audio_data.astype(np.float32) / 32768.0

                        # 处理音频
                        self.process_audio(audio_data)

                        if stream_callback:
                            stream_callback()
                    elif bytes_read < 0:
                        print(f"[音频引擎] AudioRecord 读取错误: {bytes_read}")
                        break

                except JavaException as e:
                    print(f"[音频引擎] Java异常: {e}")
                    break
                except Exception as e:
                    print(f"[音频引擎] 处理错误: {e}")

        except Exception as e:
            print(f"[音频引擎] 采集线程异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[音频引擎] Android采集线程结束")

    def _start_pyaudio_stream(self, stream_callback=None):
        """使用PyAudio作为回退方案（桌面端）"""
        try:
            import pyaudio

            self._pa = pyaudio.PyAudio()

            def callback(in_data, frame_count, time_info, status):
                if not self.is_running or self.is_paused:
                    return (None, pyaudio.paContinue)
                try:
                    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.process_audio(audio_data)
                    if stream_callback:
                        stream_callback()
                except Exception as e:
                    print(f"音频处理错误: {e}")
                return (None, pyaudio.paContinue)

            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.samplerate,
                input=True,
                frames_per_buffer=self.blocksize,
                stream_callback=callback
            )
            self._stream.start_stream()
            print(f"[音频引擎] PyAudio 已启动，采样率: {self.samplerate}")

        except ImportError:
            print("[音频引擎] 错误: 未找到音频采集库。请安装 pyaudio")
            self.is_running = False

    def stop_capture(self):
        """停止音频采集"""
        self.is_running = False

        # 停止Android AudioRecord
        if self._audio_record:
            try:
                self._audio_record.stop()
                self._audio_record.release()
            except:
                pass
            self._audio_record = None

        # 等待采集线程结束
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=1.0)

        # 停止PyAudio
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except:
                pass
            self._stream = None

        if hasattr(self, '_pa') and self._pa:
            try:
                self._pa.terminate()
            except:
                pass
            self._pa = None

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def get_latest_data(self):
        """获取最新的频谱数据（线程安全）"""
        with self.lock:
            return self.latest_freqs, self.latest_magnitude, self.latest_time_data
