"""
音频采集与FFT处理引擎 - Android修复版
使用pyjnius调用Android原生AudioRecord API

[修复] pause/resume 真正 stop/start AudioRecord，避免后台麦克风硬件持续耗电
[修复] 采集线程用 threading.Event 替代 sleep 轮询，暂停时不消耗 CPU
[修复] stop_capture 增加 release() 调用，确保硬件资源释放
"""
import numpy as np
import threading
import queue
import time
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

        # ---- 敲击(冲击)测试 ----
        # 敲击后的一段自由衰减响应包含系统的固有频率信息
        self.impact_mode = False
        self.impact_pre_sec = 0.05          # 触发前保留的音频长度
        self.impact_post_sec = 1.0          # 触发后采集时长（越长频率分辨率越高）
        self.impact_threshold_db = -26.0    # 触发电平门限（dBFS 峰值），越高越抗环境噪音
        self.impact_refractory_sec = 0.3    # 一次采集结束后的冷却时间
        self.impact_max_modes = 5           # 最多报告的模态数量
        self.impact_fmin = 20.0             # 分析下限，避开直流/工频

        self._impact_state = 'idle'         # idle / capturing / cooldown
        self._impact_prebuffer = deque(maxlen=int(samplerate * self.impact_pre_sec))
        self._impact_capture_blocks = []
        self._impact_capture_samples = 0
        self._impact_needed_samples = 0
        self._impact_ready_waveform = None
        self._impact_last_trigger = 0.0
        self._impact_noise_floor = 0.002
        self._impact_lock = threading.Lock()

        self._window_cache = {}

        self.latest_freqs = None
        self.latest_magnitude = None
        self.latest_time_data = None
        self.lock = threading.Lock()

        self._capture_thread = None
        self._stream = None

        self._audio_record = None
        self._record_thread = None
        self._is_android = False
        try:
            from jnius import autoclass
            self._is_android = True
        except ImportError:
            pass

        # [修复] 用 Event 替代 sleep 轮询，暂停时线程真正阻塞
        self._pause_event = threading.Event()
        self._pause_event.set()  # set = 非暂停; clear = 暂停

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

        self._process_impact(audio_data)

        # 敲击布防/采集期间不需要实时频谱，跳过 FFT，避免无谓占用
        if self.impact_mode:
            return None, None

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

    # ------------------------------------------------------------------
    # 敲击(冲击)测试：检测敲击 -> 截取自由衰减响应 -> 提取固有频率
    # ------------------------------------------------------------------

    def set_impact_detection(self, enabled):
        """开启/关闭敲击检测，并清空上一次的采集状态"""
        with self._impact_lock:
            self.impact_mode = bool(enabled)
            self._impact_state = 'idle'
            self._impact_prebuffer.clear()
            self._impact_capture_blocks = []
            self._impact_capture_samples = 0
            self._impact_ready_waveform = None
            self._impact_noise_floor = 0.002

    def set_impact_threshold_db(self, db):
        """触发电平门限（dBFS 峰值）：环境噪音大时调高，轻敲测不出来时调低"""
        self.impact_threshold_db = max(-80.0, min(-3.0, float(db)))

    def _impact_threshold_amp(self):
        return 10.0 ** (self.impact_threshold_db / 20.0)

    def take_impact_waveform(self):
        """取出一次已采集完成的冲击响应（取走后清空），由 UI 线程调用"""
        with self._impact_lock:
            waveform = self._impact_ready_waveform
            self._impact_ready_waveform = None
            return waveform

    def _reset_impact_capture(self):
        with self._impact_lock:
            self._impact_state = 'idle'
            self._impact_capture_blocks = []
            self._impact_capture_samples = 0
            self._impact_ready_waveform = None

    def _process_impact(self, block):
        """
        在音频线程中运行：检测敲击起始点并截取响应。
        只做触发/缓存，不做 FFT，避免阻塞音频采集。
        """
        if not self.impact_mode or len(block) == 0:
            return

        if block.ndim > 1:
            block = block[:, 0]

        block = np.asarray(block, dtype=np.float32)
        peak = float(np.max(np.abs(block)))
        rms = float(np.sqrt(np.mean(block * block)))
        now = time.monotonic()

        with self._impact_lock:
            state = self._impact_state

            if state == 'cooldown':
                if now - self._impact_last_trigger < self.impact_refractory_sec:
                    return
                state = 'idle'
                self._impact_state = 'idle'

            if state == 'capturing':
                self._impact_capture_blocks.append(block.copy())
                self._impact_capture_samples += len(block)
                if self._impact_capture_samples >= self._impact_needed_samples:
                    self._impact_ready_waveform = np.concatenate(self._impact_capture_blocks)
                    self._impact_capture_blocks = []
                    self._impact_capture_samples = 0
                    self._impact_last_trigger = now
                    self._impact_state = 'cooldown'
                return

            # idle：跟踪环境噪声底（快降慢升），避免敲击本身抬高门限
            if rms < self._impact_noise_floor:
                self._impact_noise_floor += 0.2 * (rms - self._impact_noise_floor)
            else:
                self._impact_noise_floor += 0.005 * (rms - self._impact_noise_floor)
            self._impact_noise_floor = max(self._impact_noise_floor, 1e-5)

            # 用户设定的电平门限为主，噪声底 3 倍作为兜底，避免持续噪音误触发
            threshold = max(self._impact_threshold_amp(), self._impact_noise_floor * 3.0)

            if peak < threshold:
                self._impact_prebuffer.extend(block)
                return

            # 触发：保留触发前的一小段 + 当前块
            pre = np.fromiter(self._impact_prebuffer, dtype=np.float32,
                              count=len(self._impact_prebuffer))
            self._impact_prebuffer.clear()
            self._impact_capture_blocks = [pre, block.copy()] if len(pre) else [block.copy()]
            self._impact_capture_samples = len(pre) + len(block)
            self._impact_needed_samples = int(self.samplerate * self.impact_post_sec)
            self._impact_state = 'capturing'
            print(f"[敲击] 检测到冲击 (触发峰值 {peak:.3f}, 门限 {threshold:.3f})")

    def analyze_impact(self, waveform):
        """
        分析冲击响应（纯 numpy）：
        Hann 窗 + 4 倍零填充 FFT + 对数抛物线插值细化峰值。
        """
        if waveform is None or len(waveform) < 512:
            return None

        x = np.asarray(waveform, dtype=np.float64)
        x = x - np.mean(x)
        sr = float(self.samplerate)
        n = len(x)
        n_orig = n  # 保留原始采集长度，用于确定零填充后的频率分辨率

        # 敲击响应应当自由衰减：持续环境噪音前后能量接近，直接拒绝，避免误分析
        decay_ratio = 1.0
        half = n // 2
        if half > 0:
            e_head = float(np.mean(x[:half] * x[:half]))
            e_tail = float(np.mean(x[half:] * x[half:]))
            if e_tail > 0 and e_head < e_tail * 1.3:
                print("[敲击] 响应无明显衰减，判定为环境噪音，已丢弃")
                return None
            if e_tail > 0:
                decay_ratio = e_head / e_tail

        # 自适应截取：敲击往往 0.1~0.3 秒就衰减完，若用固定 1 秒窗口做 FFT，
        # 幅值会被平均到后面的静音段，被稀释几十~几百倍（衰减越快越严重），
        # 导致波形看起来极小、快衰减模态被底噪淹没。这里只保留响应真正存在的
        # 段落，尾部静音截掉，再靠零填充维持频率分辨率。
        if n > int(sr * 0.15):
            env_win = max(1, int(sr * 0.01))
            env = np.convolve(np.abs(x), np.ones(env_win) / env_win, mode='same')
            pk_idx = int(np.argmax(env))
            peak_env = float(env[pk_idx])
            tail_len = min(len(env), int(sr * 0.03))
            noise = float(np.median(env[-tail_len:])) if tail_len > 0 else 0.0
            cutoff = max(peak_env * 0.04, noise * 2.0, 1e-4)
            below = env[pk_idx:] < cutoff
            trunc = n
            if np.any(below):
                need = max(1, int(sr * 0.04))  # 连续 40ms 低于门限才确认衰减结束
                run = 0
                for j in range(len(below)):
                    run = run + 1 if below[j] else 0
                    if run >= need:
                        trunc = pk_idx + j - need + 1
                        break
            trunc = min(n, max(trunc, int(sr * 0.08)))  # 至少保留 80ms，保证分辨率
            if trunc < n:
                x = x[:trunc]
                n = trunc

        window = np.hanning(n)
        wsum = float(np.sum(window))
        if wsum <= 0:
            return None

        # 零填充到原始采集长度的 4 倍，频率分辨率与未截断时一致（df≈0.18Hz）
        nfft = 1 << int(np.ceil(np.log2(max(n_orig * 4, 8192))))
        spectrum = np.abs(np.fft.rfft(x * window, nfft)) * 2.0 / wsum
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        df = sr / float(nfft)

        f_min = max(float(self.impact_fmin), float(self.freq_min))
        f_max = min(float(self.freq_max) if self.freq_max > 0 else sr / 2.0, sr / 2.0)
        if f_max <= f_min:
            f_max = sr / 2.0

        band = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(band):
            return None

        band_idx = np.where(band)[0]
        band_spec = spectrum[band_idx]
        max_amp = float(np.max(band_spec))
        if max_amp < 1e-9:
            return None

        left = band_spec[1:-1] > band_spec[:-2]
        right = band_spec[1:-1] > band_spec[2:]
        candidates = band_idx[1:-1][left & right]
        candidates = candidates[spectrum[candidates] >= max_amp * 0.02]
        if len(candidates) > 2000:
            candidates = candidates[np.argsort(spectrum[candidates])[-2000:]]
        if len(candidates) == 0:
            return None

        # 按幅度从大到小贪心保留，峰值间隔至少 ~4Hz，避免毛刺
        min_sep = max(1, int(round(4.0 / df)))
        kept = []
        for idx in candidates[np.argsort(spectrum[candidates])[::-1]]:
            if all(abs(int(idx) - k) >= min_sep for k in kept):
                kept.append(int(idx))
                if len(kept) >= self.impact_max_modes * 4:
                    break
        kept.sort()

        modes = []
        mode_idx = []
        for i in kept:
            if i <= 0 or i >= len(spectrum) - 1:
                continue
            y0 = np.log(spectrum[i - 1] + 1e-15)
            y1 = np.log(spectrum[i] + 1e-15)
            y2 = np.log(spectrum[i + 1] + 1e-15)
            denom = y0 - 2.0 * y1 + y2
            delta = 0.0 if abs(denom) < 1e-12 else 0.5 * (y0 - y2) / denom
            delta = max(-0.5, min(0.5, delta))
            f_ref = float(freqs[i] + delta * df)
            if f_ref < f_min or f_ref > f_max:
                continue
            amp_ref = float(np.exp(y1 - 0.25 * (y0 - y2) * delta))
            modes.append((f_ref, amp_ref))
            mode_idx.append(int(i))

        if not modes:
            return None

        # 主模态：按幅度取最大值，并记录其在全谱中的索引，用于计算峰质量指标
        amps = np.array([a for _, a in modes], dtype=np.float64)
        dom_pos = int(np.argmax(amps))
        dom_i = mode_idx[dom_pos]
        f0 = float(modes[dom_pos][0])
        peak_amp = float(spectrum[dom_i])

        # ---- 精度置信度指标（单次敲击）----
        # 1) 信噪比：主峰相对频带内噪声中位数（中位数对离散峰不敏感，稳健）
        noise_floor = float(np.median(band_spec))
        snr_db = 20.0 * np.log10(peak_amp / max(noise_floor, 1e-12))

        # 2) -3dB 带宽（幅度降到 1/√2），用于衡量峰锐度/频率可分辨性
        half_amp = peak_amp / np.sqrt(2.0)
        li = dom_i
        while li > 0 and spectrum[li] > half_amp:
            li -= 1
        ri = dom_i
        while ri < len(spectrum) - 1 and spectrum[ri] > half_amp:
            ri += 1
        bandwidth = max(df, (ri - li) * df)
        q_factor = f0 / bandwidth if bandwidth > 0 else 0.0

        # 3) 与次峰的分离度：主峰与距离较远的最大竞争峰之比（dB）
        guard = max(1, int(round(8.0 / df)))
        others = [spectrum[i] for i in kept if abs(i - dom_i) >= guard]
        sep_db = 20.0 * np.log10(peak_amp / max(max(others), 1e-12)) if others else 60.0

        # 4) 窗口内周期数：低频模态在短窗里周期少，频率估计相对不稳
        duration_eff = n / sr
        cycles = duration_eff * f0

        # 合成精度分：SNR 权重最高，其次峰分离度、衰减幅度、周期数
        clip01 = lambda v: max(0.0, min(1.0, v))
        s_snr = clip01((snr_db - 10.0) / 30.0)
        s_sep = clip01((sep_db - 6.0) / 24.0)
        s_decay = clip01(np.log10(max(decay_ratio, 1.0)) / 1.5)
        s_res = clip01((cycles - 3.0) / 12.0)
        precision = 100.0 * (0.4 * s_snr + 0.25 * s_sep + 0.2 * s_decay + 0.15 * s_res)

        modes.sort(key=lambda m: m[1], reverse=True)
        dominant = modes[0]
        modes = sorted(modes[:self.impact_max_modes], key=lambda m: m[0])

        return {
            'freqs': freqs,
            'magnitude': spectrum,
            'modes': modes,
            'dominant': dominant,
            'duration': duration_eff,
            'metrics': {
                'snr_db': float(snr_db),
                'bandwidth': float(bandwidth),
                'q_factor': float(q_factor),
                'sep_db': float(sep_db),
                'decay_ratio': float(decay_ratio),
                'cycles': float(cycles),
                'precision': float(precision),
            },
        }

    def start_capture(self, stream_callback=None):
        """开始音频采集"""
        self.is_running = True
        self.is_paused = False
        self._pause_event.set()
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

            min_buffer_size = AudioRecord.getMinBufferSize(
                self.samplerate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            if min_buffer_size < 0:
                print(f"[音频引擎] AudioRecord.getMinBufferSize 失败: {min_buffer_size}")
                self.is_running = False
                return

            buffer_size = max(min_buffer_size, self.blocksize * 2)

            self._audio_record = AudioRecord(
                AudioSource.MIC,
                self.samplerate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                buffer_size
            )

            if self._audio_record.getState() != AudioRecord.STATE_INITIALIZED:
                print("[音频引擎] AudioRecord 初始化失败，请检查麦克风权限")
                self._audio_record.release()
                self._audio_record = None
                self.is_running = False
                return

            self._audio_record.startRecording()
            print(f"[音频引擎] Android AudioRecord 已启动，采样率: {self.samplerate}")

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
        """Android音频采集线程 - [修复] 用 Event 替代 sleep 轮询"""
        try:
            from jnius import JavaException
            import time

            audio_buffer = bytearray(buffer_size)

            while self.is_running:
                # [修复] 用 Event.wait 替代 sleep(0.05) 轮询
                # 暂停时线程真正阻塞，不消耗 CPU，不阻止系统休眠
                if not self._pause_event.wait(timeout=1.0):
                    # 1 秒内没收到恢复信号，继续等待
                    continue

                if not self.is_running:
                    break

                try:
                    bytes_read = self._audio_record.read(audio_buffer, 0, buffer_size)

                    if bytes_read > 0:
                        audio_data = np.frombuffer(bytes(audio_buffer[:bytes_read]), dtype=np.int16)
                        audio_data = audio_data.astype(np.float32) / 32768.0

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
        # [修复] 唤醒可能阻塞在 Event.wait 上的线程
        self._pause_event.set()
        # 丢弃未完成的敲击采集
        self._reset_impact_capture()

        # [修复] 先 join 线程，再释放 AudioRecord（避免线程还在 read 时被 release）
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=3.0)

        # [修复] 确保同时调用 stop() 和 release()
        if self._audio_record:
            try:
                self._audio_record.stop()
            except:
                pass
            try:
                self._audio_record.release()
            except:
                pass
            self._audio_record = None

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
        """[修复] 暂停采集 — 真正停止 AudioRecord 硬件，不仅设标志位"""
        self.is_paused = True
        self._pause_event.clear()
        # 暂停时丢弃未完成的敲击采集，恢复后重新触发
        self._reset_impact_capture()

        # 停止 Android AudioRecord 硬件，释放麦克风供电
        if self._is_android and self._audio_record:
            try:
                self._audio_record.stop()
                print("[音频引擎] AudioRecord 已停止（暂停），麦克风硬件已释放")
            except Exception as e:
                print(f"[音频引擎] 暂停时 stop 失败: {e}")

    def resume(self):
        """[修复] 恢复采集 — 重新启动 AudioRecord 硬件"""
        self.is_paused = False
        self._pause_event.set()

        # 重新启动 Android AudioRecord
        if self._is_android and self._audio_record:
            try:
                self._audio_record.startRecording()
                print("[音频引擎] AudioRecord 已恢复录音")
            except Exception as e:
                print(f"[音频引擎] 恢复时 startRecording 失败: {e}")
                # 如果重启失败，尝试完全重新初始化
                try:
                    self._audio_record.release()
                except:
                    pass
                self._audio_record = None
                self._start_android_audio()

    def get_latest_data(self):
        """获取最新的频谱数据（线程安全）"""
        with self.lock:
            return self.latest_freqs, self.latest_magnitude, self.latest_time_data
