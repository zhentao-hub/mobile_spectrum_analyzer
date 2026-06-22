"""
Mobile Spectrum Analyzer - Kivy Version
Supports real-time microphone capture + FFT spectrum display
"""
import os
import sys
import time
import numpy as np

# Kivy 配置
os.environ['KIVY_NO_CONSOLELOG'] = '0'
os.environ['KIVY_WINDOW'] = 'sdl2'

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Line, Rectangle, Ellipse
from kivy.graphics.vertex_instructions import Mesh
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import NumericProperty, ListProperty, StringProperty, BooleanProperty
from kivy.metrics import dp

from audio_engine import AudioSpectrumEngine


class SpectrumCanvas(FloatLayout):
    """自定义频谱绘制画布"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freqs = None
        self.magnitude = None
        self.time_data = None
        self.peaks = []
        self.show_time_domain = False
        self.log_scale = False
        self.amp_max = 0.5
        self.paused = False
        self.samplerate = 48000

        self.zoom_start = 0.0
        self.zoom_end = 1.0

        self.dragging = False
        self.last_touch_x = 0

        self.grid_color = (0.3, 0.3, 0.3, 1)
        self.spectrum_color = (0, 1, 1, 1)
        self.peak_color = (1, 0, 0, 1)
        self.time_color = (1, 1, 0, 1)
        self.text_color = (1, 1, 1, 1)

        self.grid_lines = []
        self.spectrum_line = None
        self.time_line = None
        self.peak_labels = []

        self.axis_labels = []
        self.axis_titles = []
        self.peak_text_labels = []

        self.crosshair_line = None
        self.cursor_info_label = None

        self._last_peak_color = None

        with self.canvas:
            Color(0.1, 0.1, 0.1, 1)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)

            self.grid_color_inst = Color(*self.grid_color)
            for _ in range(20):
                self.grid_lines.append(Line(points=[], width=1))

            self.spectrum_color_inst = Color(*self.spectrum_color)
            self.spectrum_line = Line(points=[], width=2)

            self.time_color_inst = Color(*self.time_color)
            self.time_line = Line(points=[], width=1.5)

            self.peak_color_inst = Color(*self.peak_color)
            self.peak_ellipses = []
            self.peak_texts = []

            Color(1, 1, 0, 0.6)
            self.crosshair_line = Line(points=[], width=1)

        self.label_layout = FloatLayout(pos=self.pos, size=self.size)
        self.add_widget(self.label_layout)

        self.peak_label_layout = FloatLayout(pos=self.pos, size=self.size)
        self.add_widget(self.peak_label_layout)

        self.cursor_info_label = Label(
            text='',
            font_size='12sp',
            color=(1, 1, 0, 1),
            size_hint=(None, None),
            size=(180, 40),
            halign='left',
            valign='top',
            markup=True
        )
        self.peak_label_layout.add_widget(self.cursor_info_label)

        self._create_axis_labels()

        self.bind(pos=self._update_canvas, size=self._update_canvas)
        self.bind(pos=self._update_peak_layout, size=self._update_peak_layout)
        Window.bind(mouse_pos=self._on_mouse_pos)

    def cleanup(self):
        Window.unbind(mouse_pos=self._on_mouse_pos)

    def _on_mouse_pos(self, window, pos):
        touch_x, touch_y = self.to_widget(*pos)
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()
        in_plot = (plot_x <= touch_x <= plot_x + plot_w and plot_y <= touch_y <= plot_y + plot_h)
        if not in_plot or self.show_time_domain:
            self._hide_cursor()
            return
        self._update_cursor(touch_x, touch_y)

    def _update_peak_layout(self, *args):
        self.peak_label_layout.pos = self.pos
        self.peak_label_layout.size = self.size

    def _create_axis_labels(self):
        for label in self.axis_labels + self.axis_titles:
            if label in self.label_layout.children:
                self.label_layout.remove_widget(label)
        self.axis_labels = []
        self.axis_titles = []

        x_title = Label(
            text='Frequency (Hz)',
            font_size='12sp',
            color=(0.5, 0.8, 1, 1),
            size_hint=(None, None),
            size=(120, 20)
        )
        self.axis_titles.append(x_title)
        self.label_layout.add_widget(x_title)

        y_title = Label(
            text='Magnitude',
            font_size='12sp',
            color=(0.5, 0.8, 1, 1),
            size_hint=(None, None),
            size=(100, 20)
        )
        self.axis_titles.append(y_title)
        self.label_layout.add_widget(y_title)

        for i in range(6):
            label = Label(
                text='',
                font_size='11sp',
                color=self.text_color,
                size_hint=(None, None),
                size=(80, 20)
            )
            self.axis_labels.append(label)
            self.label_layout.add_widget(label)

        for i in range(6):
            label = Label(
                text='',
                font_size='11sp',
                color=self.text_color,
                size_hint=(None, None),
                size=(70, 20),
                halign='right'
            )
            self.axis_labels.append(label)
            self.label_layout.add_widget(label)

    def on_touch_down(self, touch):
        if not self.paused or self.show_time_domain:
            return super().on_touch_down(touch)

        touch_x, touch_y = self.to_widget(*touch.pos)
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()
        if not (plot_x <= touch_x <= plot_x + plot_w and plot_y <= touch_y <= plot_y + plot_h):
            return super().on_touch_down(touch)

        if touch.button == 'scrollup':
            zoom_factor = 0.9
        elif touch.button == 'scrolldown':
            zoom_factor = 1.1
        elif touch.button == 'left':
            self.dragging = True
            self.last_touch_x = touch_x
            return True
        else:
            return super().on_touch_down(touch)

        mouse_ratio = (touch_x - plot_x) / plot_w if plot_w > 0 else 0.5
        current_range = self.zoom_end - self.zoom_start
        new_range = max(0.05, min(1.0, current_range * zoom_factor))
        center = self.zoom_start + current_range * mouse_ratio
        new_start = max(0.0, center - new_range * mouse_ratio)
        new_end = min(1.0, new_start + new_range)

        if new_end >= 1.0:
            new_end = 1.0
            new_start = max(0.0, new_end - new_range)

        self.zoom_start = new_start
        self.zoom_end = new_end
        self.draw()
        return True

    def on_touch_move(self, touch):
        touch_x, touch_y = self.to_widget(*touch.pos)
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()
        in_plot = (plot_x <= touch_x <= plot_x + plot_w and plot_y <= touch_y <= plot_y + plot_h)

        if not in_plot:
            self._hide_cursor()
            if self.dragging:
                self.dragging = False
            return super().on_touch_move(touch)

        if self.paused and not self.show_time_domain and self.dragging:
            if plot_w <= 0:
                return super().on_touch_move(touch)

            dx = touch_x - self.last_touch_x
            offset = -dx / plot_w * (self.zoom_end - self.zoom_start)
            new_start = max(0.0, min(1.0 - (self.zoom_end - self.zoom_start), self.zoom_start + offset))
            new_end = new_start + (self.zoom_end - self.zoom_start)
            self.zoom_start = new_start
            self.zoom_end = min(1.0, new_end)
            self.last_touch_x = touch_x
            self.draw()
            return True

        if not self.show_time_domain:
            self._update_cursor(touch_x, touch_y)
            return True

        return super().on_touch_move(touch)

    def _update_cursor(self, touch_x, touch_y):
        if self.freqs is None or self.magnitude is None or len(self.freqs) == 0:
            self._hide_cursor()
            return

        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()
        if plot_w <= 0:
            self._hide_cursor()
            return

        mouse_ratio = (touch_x - plot_x) / plot_w
        f_min_all, f_max_all = self.freqs[0], self.freqs[-1]
        freq_range_all = f_max_all - f_min_all if f_max_all > f_min_all else 1

        if self.paused:
            f_min = f_min_all + freq_range_all * self.zoom_start
            f_max = f_min_all + freq_range_all * self.zoom_end
        else:
            f_min = f_min_all
            f_max = f_max_all

        freq = f_min + (f_max - f_min) * mouse_ratio
        idx = np.argmin(np.abs(self.freqs - freq))
        actual_freq = self.freqs[idx]
        actual_mag = self.magnitude[idx]

        is_peak = False
        if self.peaks and len(self.peaks) > 0:
            nearest_peak = min(self.peaks, key=lambda p: abs(float(p[0]) - freq))
            peak_freq = float(nearest_peak[0])
            peak_mag = float(nearest_peak[1])
            threshold = max((f_max - f_min) * 0.03, 20)
            if abs(peak_freq - freq) < threshold:
                actual_freq = peak_freq
                actual_mag = peak_mag
                is_peak = True
                idx = np.argmin(np.abs(self.freqs - peak_freq))

        px = plot_x + plot_w * (actual_freq - f_min) / (f_max - f_min) if (f_max > f_min) else plot_x

        self.crosshair_line.points = [float(px), float(plot_y), float(px), float(plot_y + plot_h)]

        if self.log_scale:
            mag_text = f'{actual_mag:.1f} dB'
        else:
            mag_text = f'{actual_mag:.4f}'

        peak_marker = ' [color=ffaa00](Peak)[/color]' if is_peak else ''
        self.cursor_info_label.text = f'[b]{actual_freq:.0f} Hz{peak_marker}\n{mag_text}[/b]'

        info_w, info_h = 180, 40
        label_x = min(px + 10, plot_x + plot_w - info_w)
        label_y = max(touch_y - 10, plot_y)
        self.cursor_info_label.pos = (float(label_x), float(label_y))

    def _hide_cursor(self):
        self.crosshair_line.points = []
        self.cursor_info_label.text = ''

    def on_touch_up(self, touch):
        if self.dragging:
            self.dragging = False
            return True
        return super().on_touch_up(touch)

    def _update_canvas(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        self.draw()

    def update_data(self, freqs, magnitude, time_data, peaks=None):
        self.freqs = freqs
        self.magnitude = magnitude
        self.time_data = time_data
        self.peaks = peaks or []
        self.draw()

    def draw(self):
        try:
            if self.show_time_domain:
                self._draw_time_domain()
            else:
                self._draw_frequency_domain()
            self._draw_grid()
        except Exception as e:
            print(f"[错误] draw 异常: {e}")
            import traceback
            traceback.print_exc()

    def _draw_grid(self):
        x, y = self.pos
        w, h = self.size

        left_margin = dp(60)
        bottom_margin = dp(30)
        right_margin = dp(10)
        top_margin = dp(10)

        plot_x = x + left_margin
        plot_y = y + bottom_margin
        plot_w = w - left_margin - right_margin
        plot_h = h - bottom_margin - top_margin

        for i, line in enumerate(self.grid_lines[:5]):
            gy = plot_y + plot_h * (i + 1) / 6
            line.points = [plot_x, gy, plot_x + plot_w, gy]

        for i, line in enumerate(self.grid_lines[5:10]):
            gx = plot_x + plot_w * (i + 1) / 6
            line.points = [gx, plot_y, gx, plot_y + plot_h]

        for line in self.grid_lines[10:]:
            line.points = []

    def _get_plot_area(self):
        x, y = self.pos
        w, h = self.size
        left_margin = dp(60)
        bottom_margin = dp(30)
        right_margin = dp(10)
        top_margin = dp(10)
        return (
            x + left_margin,
            y + bottom_margin,
            w - left_margin - right_margin,
            h - bottom_margin - top_margin
        )

    def _draw_frequency_domain(self):
        if self.freqs is None or self.magnitude is None or len(self.freqs) == 0:
            self.spectrum_line.points = []
            self._clear_peak_labels()
            self._update_axis_labels(0, 20000, 0, 1, x_title='Frequency (Hz)', y_title='Magnitude', x_fmt='freq', y_fmt='mag')
            return

        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()

        f_min_all, f_max_all = self.freqs[0], self.freqs[-1]
        freq_range_all = f_max_all - f_min_all if f_max_all > f_min_all else 1

        if self.paused:
            f_min = f_min_all + freq_range_all * self.zoom_start
            f_max = f_min_all + freq_range_all * self.zoom_end
        else:
            f_min = f_min_all
            f_max = f_max_all

        mask = (self.freqs >= f_min) & (self.freqs <= f_max)
        display_freqs = self.freqs[mask]
        display_mags = self.magnitude[mask]

        if len(display_freqs) == 0:
            self.spectrum_line.points = []
            self._clear_peak_labels()
            self._update_axis_labels(f_min, f_max, 0, 1, x_title='Frequency (Hz)', y_title='Magnitude', x_fmt='freq', y_fmt='mag')
            return

        freq_range = f_max - f_min if f_max > f_min else 1

        if self.log_scale:
            mag_min = max(-120, np.min(display_mags) - 10)
            mag_max = max(0, np.max(display_mags) + 10)
        else:
            mag_min = 0
            mag_max = max(self.amp_max, 0.001)

        mag_range = mag_max - mag_min if mag_max > mag_min else 1

        points = []
        n = len(display_freqs)
        max_points = min(n, 800)
        step = max(1, n // max_points)

        for i in range(0, n, step):
            fx = plot_x + plot_w * (display_freqs[i] - f_min) / freq_range
            if self.log_scale:
                fy = plot_y + plot_h * (display_mags[i] - mag_min) / mag_range
            else:
                fy = plot_y + plot_h * (display_mags[i] - mag_min) / mag_range
            fy = max(plot_y, min(plot_y + plot_h, fy))
            points.extend([fx, fy])

        self.spectrum_line.points = points
        self.time_line.points = []

        self._update_axis_labels(f_min, f_max, mag_min, mag_max, x_title='Frequency (Hz)', y_title='Magnitude', x_fmt='freq', y_fmt='mag')

        display_peaks = [(f, m) for f, m in self.peaks if f_min <= f <= f_max]
        self._draw_peaks(f_min, freq_range, mag_min, mag_range, plot_x, plot_y, plot_w, plot_h, display_peaks)

    def _update_axis_labels(self, x_min, x_max, y_min, y_max, x_title='Frequency (Hz)', y_title='Magnitude', x_fmt='freq', y_fmt='mag'):
        x, y = self.pos
        w, h = self.size

        left_margin = dp(60)
        bottom_margin = dp(30)
        right_margin = dp(10)
        top_margin = dp(10)

        plot_x = x + left_margin
        plot_y = y + bottom_margin
        plot_w = w - left_margin - right_margin
        plot_h = h - bottom_margin - top_margin

        if self.axis_titles:
            self.axis_titles[0].text = x_title
            self.axis_titles[0].pos = (float(plot_x + plot_w / 2 - 60), float(y - 5))
            self.axis_titles[1].text = y_title
            self.axis_titles[1].pos = (float(x - 5), float(plot_y + plot_h / 2 - 10))

        for i in range(6):
            label = self.axis_labels[i]
            x_val = float(x_min + (x_max - x_min) * i / 5)
            if x_fmt == 'freq':
                if x_val >= 1000:
                    label.text = f'{x_val/1000:.1f}k'
                else:
                    label.text = f'{x_val:.0f}'
            elif x_fmt == 'time_ms':
                if x_val >= 1000:
                    label.text = f'{x_val/1000:.2f}s'
                else:
                    label.text = f'{x_val:.1f}ms'
            else:
                label.text = f'{x_val:.1f}'
            label.pos = (float(plot_x + plot_w * i / 5 - 40), float(y + 5))

        for i in range(6):
            label = self.axis_labels[6 + i]
            y_val = float(y_min + (y_max - y_min) * i / 5)
            if y_fmt == 'mag':
                if self.log_scale:
                    label.text = f'{y_val:.0f}dB'
                else:
                    y_range = y_max - y_min
                    if y_range >= 1:
                        label.text = f'{y_val:.2f}'
                    elif y_range >= 0.1:
                        label.text = f'{y_val:.3f}'
                    else:
                        label.text = f'{y_val:.4f}'
            elif y_fmt == 'amp_norm':
                label.text = f'{y_val:.1f}'
            else:
                label.text = f'{y_val:.2f}'
            label.pos = (float(x + 5), float(plot_y + plot_h * i / 5 - 10))

    def _draw_time_domain(self):
        self._clear_peak_labels()

        if self.time_data is None or len(self.time_data) == 0:
            self.time_line.points = []
            self._update_axis_labels(0, 100, -1.0, 1.0, x_title='Time (ms)', y_title='Amplitude', x_fmt='time_ms', y_fmt='amp_norm')
            return

        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()

        data = self.time_data
        n = len(data)
        max_points = min(n, 800)
        step = max(1, n // max_points)

        data_max = np.max(np.abs(data))
        if data_max > 0:
            data = data / data_max
        else:
            data = np.zeros_like(data)

        points = []
        for i in range(0, n, step):
            tx = plot_x + plot_w * i / n
            ty = plot_y + plot_h * 0.5 + plot_h * 0.45 * data[i]
            points.extend([tx, ty])

        self.time_line.points = points
        self.spectrum_line.points = []

        duration_ms = n / self.samplerate * 1000
        self._update_axis_labels(0, duration_ms, -1.0, 1.0, x_title='Time (ms)', y_title='Amplitude', x_fmt='time_ms', y_fmt='amp_norm')

    def _clear_peak_labels(self):
        for ellipse in self.peak_ellipses:
            if ellipse in self.canvas.children:
                self.canvas.remove(ellipse)
        self.peak_ellipses = []

        if self._last_peak_color and self._last_peak_color in self.canvas.children:
            self.canvas.remove(self._last_peak_color)
        self._last_peak_color = None

        for widget in list(self.peak_label_layout.children):
            if widget is not self.cursor_info_label:
                self.peak_label_layout.remove_widget(widget)
        self.peak_text_labels = []

        for line in getattr(self, 'peak_lines', []):
            if line in self.canvas.children:
                self.canvas.remove(line)
        self.peak_lines = []

    def _draw_peaks(self, f_min, freq_range, mag_min, mag_range, plot_x=None, plot_y=None, plot_w=None, plot_h=None, peaks=None):
        self._clear_peak_labels()

        if peaks is None:
            peaks = self.peaks

        if not peaks:
            return

        if plot_x is None:
            plot_x, plot_y, plot_w, plot_h = self._get_plot_area()

        sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

        peak_positions = []
        for freq, mag in sorted_peaks:
            px = plot_x + plot_w * (float(freq) - f_min) / freq_range
            if self.log_scale:
                py = plot_y + plot_h * (float(mag) - mag_min) / mag_range
            else:
                py = plot_y + plot_h * (float(mag) - mag_min) / mag_range
            py = max(plot_y, min(plot_y + plot_h, py))
            peak_positions.append((px, py, freq, mag))

        with self.canvas:
            self._last_peak_color = Color(*self.peak_color)
            for px, py, freq, mag in peak_positions:
                ellipse = Ellipse(pos=(float(px - 4), float(py - 4)), size=(8, 8))
                self.peak_ellipses.append(ellipse)

        for i, (px, py, freq, mag) in enumerate(peak_positions):
            label_x = min(px + dp(8), plot_x + plot_w - dp(100))
            label_y = py + dp(8) + i * dp(16)
            label_y = min(label_y, plot_y + plot_h - dp(10))

            peak_label = Label(
                text=f'{float(freq):.0f}Hz',
                font_size='11sp',
                color=(1, 0.9, 0.3, 1),
                size_hint=(None, None),
                size=(100, 20),
                halign='left'
            )
            peak_label.pos = (float(label_x), float(label_y))
            self.peak_text_labels.append(peak_label)
            self.peak_label_layout.add_widget(peak_label)


class ControlPanel(ScrollView):
    """控制面板（可滚动）"""

    def __init__(self, app_ref, **kwargs):
        super().__init__(**kwargs)
        self.app_ref = app_ref
        self.size_hint_x = None
        self.width = dp(280)
        self.bar_width = dp(10)
        self.scroll_type = ['bars', 'content']

        self.layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(900),
            padding=dp(10),
            spacing=dp(8)
        )
        self.layout.bind(minimum_height=self.layout.setter('height'))
        self.add_widget(self.layout)

        self.layout.add_widget(Label(
            text='[b]Spectrum Control[/b]',
            markup=True,
            size_hint_y=None,
            height=dp(30),
            color=(1, 1, 1, 1)
        ))

        btn_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(60), spacing=dp(10))
        self.start_btn = Button(
            text='▶ Start',
            on_press=self.on_start,
            font_size='18sp',
            background_color=(0.2, 0.7, 0.2, 1),
            color=(1, 1, 1, 1),
            size_hint=(1, None),
            height=dp(50)
        )
        self.stop_btn = Button(
            text='⏹ Stop',
            on_press=self.on_stop,
            disabled=True,
            font_size='18sp',
            background_color=(0.7, 0.2, 0.2, 1),
            color=(1, 1, 1, 1),
            size_hint=(1, None),
            height=dp(50)
        )
        btn_layout.add_widget(self.start_btn)
        btn_layout.add_widget(self.stop_btn)
        self.layout.add_widget(btn_layout)

        self.pause_btn = ToggleButton(
            text='⏸ Pause',
            size_hint_y=None,
            height=dp(50),
            on_press=self.on_pause,
            disabled=True,
            font_size='18sp',
            background_color=(0.7, 0.5, 0.2, 1),
            color=(1, 1, 1, 1),
        )
        self.layout.add_widget(self.pause_btn)

        self.layout.add_widget(Label(text='Display Mode:', size_hint_y=None, height=dp(25), color=(0.8, 0.8, 0.8, 1)))
        mode_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.freq_mode_btn = ToggleButton(text='Frequency', group='mode', state='down')
        self.time_mode_btn = ToggleButton(text='Time', group='mode')
        self.freq_mode_btn.bind(on_press=lambda x: self.on_mode_change(False))
        self.time_mode_btn.bind(on_press=lambda x: self.on_mode_change(True))
        mode_layout.add_widget(self.freq_mode_btn)
        mode_layout.add_widget(self.time_mode_btn)
        self.layout.add_widget(mode_layout)

        self.layout.add_widget(Label(text='FFT Size:', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        self.fft_slider = Slider(min=8, max=14, value=11, step=1, size_hint_y=None, height=dp(40))
        self.fft_label = Label(text='2048', size_hint_y=None, height=dp(20), color=(1, 1, 1, 1))
        self.fft_slider.bind(value=self.on_fft_change)
        self.layout.add_widget(self.fft_slider)
        self.layout.add_widget(self.fft_label)

        self.layout.add_widget(Label(text='Window:', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        window_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.window_btns = {}
        for wname in ['hann', 'hamming', 'blackman', 'rect']:
            btn = ToggleButton(
                text=wname,
                group='window',
                state='down' if wname == 'hann' else 'normal'
            )
            btn.bind(on_press=lambda x, n=wname: self.on_window_change(n))
            self.window_btns[wname] = btn
            window_layout.add_widget(btn)
        self.layout.add_widget(window_layout)

        self.log_btn = ToggleButton(
            text='Log Scale (dB)',
            size_hint_y=None,
            height=dp(40)
        )
        self.log_btn.bind(on_press=self.on_log_change)
        self.layout.add_widget(self.log_btn)

        self.dc_btn = ToggleButton(
            text='Remove DC',
            state='down',
            size_hint_y=None,
            height=dp(40)
        )
        self.dc_btn.bind(on_press=self.on_dc_change)
        self.layout.add_widget(self.dc_btn)

        self.avg_btn = ToggleButton(
            text='Spectrum Avg',
            size_hint_y=None,
            height=dp(40)
        )
        self.avg_btn.bind(on_press=self.on_avg_change)
        self.layout.add_widget(self.avg_btn)

        self.layout.add_widget(Label(text='Peak Count:', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        self.peak_slider = Slider(min=0, max=5, value=3, step=1, size_hint_y=None, height=dp(40))
        self.peak_label = Label(text='3', size_hint_y=None, height=dp(20), color=(1, 1, 1, 1))
        self.peak_slider.bind(value=self.on_peak_change)
        self.layout.add_widget(self.peak_slider)
        self.layout.add_widget(self.peak_label)

        self.layout.add_widget(Label(text='Min Frequency (Hz):', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        self.freq_min_slider = Slider(min=0, max=5000, value=0, step=100, size_hint_y=None, height=dp(40))
        self.freq_min_label = Label(text='0 Hz', size_hint_y=None, height=dp(20), color=(1, 1, 1, 1))
        self.freq_min_slider.bind(value=self.on_freq_min_change)
        self.layout.add_widget(self.freq_min_slider)
        self.layout.add_widget(self.freq_min_label)

        self.layout.add_widget(Label(text='Max Frequency (Hz):', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        self.freq_max_slider = Slider(min=100, max=24000, value=20000, step=100, size_hint_y=None, height=dp(40))
        self.freq_max_label = Label(text='20000 Hz', size_hint_y=None, height=dp(20), color=(1, 1, 1, 1))
        self.freq_max_slider.bind(value=self.on_freq_max_change)
        self.layout.add_widget(self.freq_max_slider)
        self.layout.add_widget(self.freq_max_label)

        self.layout.add_widget(Label(text='Max Magnitude:', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        self.amp_max_slider = Slider(min=0.001, max=0.5, value=0.5, step=0.001, size_hint_y=None, height=dp(40))
        self.amp_max_label = Label(text='0.500', size_hint_y=None, height=dp(20), color=(1, 1, 1, 1))
        self.amp_max_slider.bind(value=self.on_amp_max_change)
        self.layout.add_widget(self.amp_max_slider)
        self.layout.add_widget(self.amp_max_label)

        self.layout.add_widget(Label())

    def on_start(self, instance):
        self.app_ref.start_capture()
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.pause_btn.disabled = False

    def on_stop(self, instance):
        self.app_ref.stop_capture()
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        self.pause_btn.state = 'normal'
        self.pause_btn.disabled = True

    def on_pause(self, instance):
        if instance.state == 'down':
            self.app_ref.pause_capture()
        else:
            self.app_ref.resume_capture()

    def on_mode_change(self, time_domain):
        self.app_ref.set_display_mode(time_domain)

    def on_fft_change(self, instance, value):
        fft_size = int(2 ** value)
        self.fft_label.text = str(fft_size)
        self.app_ref.set_fft_size(fft_size)

    def on_window_change(self, window_type):
        self.app_ref.set_window_type(window_type)

    def on_log_change(self, instance):
        self.app_ref.set_log_scale(instance.state == 'down')

    def on_dc_change(self, instance):
        self.app_ref.set_dc_remove(instance.state == 'down')

    def on_avg_change(self, instance):
        self.app_ref.set_spectrum_avg(instance.state == 'down')

    def on_peak_change(self, instance, value):
        self.peak_label.text = str(int(value))
        self.app_ref.set_peak_count(int(value))

    def on_freq_min_change(self, instance, value):
        self.freq_min_label.text = f'{int(value)} Hz'
        self.app_ref.set_freq_min(int(value))

    def on_freq_max_change(self, instance, value):
        self.freq_max_label.text = f'{int(value)} Hz'
        self.app_ref.set_freq_max(int(value))

    def on_amp_max_change(self, instance, value):
        self.amp_max_label.text = f'{value:.3f}'
        self.app_ref.set_amp_max(value)


class InfoPanel(BoxLayout):
    """信息显示面板"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = dp(30)
        self.padding = dp(5)

        self.status_label = Label(
            text='Status: Ready',
            color=(0.8, 1, 0.8, 1),
            size_hint_x=0.3
        )
        self.peak_label = Label(
            text='Peaks: --',
            color=(1, 0.8, 0.8, 1),
            size_hint_x=0.4
        )
        self.fps_label = Label(
            text='FPS: --',
            color=(0.8, 0.8, 1, 1),
            size_hint_x=0.3
        )

        self.add_widget(self.status_label)
        self.add_widget(self.peak_label)
        self.add_widget(self.fps_label)

    def update_status(self, text):
        self.status_label.text = f'Status: {text}'

    def update_peaks(self, peaks):
        if peaks:
            text = ' | '.join([f'{f:.0f}Hz' for f, m in peaks])
            self.peak_label.text = f'Peaks: {text}'
        else:
            self.peak_label.text = 'Peaks: --'

    def update_fps(self, fps):
        self.fps_label.text = f'FPS: {fps:.1f}'


class SpectrumAnalyzerApp(App):
    """主应用类"""

    def build(self):
        Window.clearcolor = (0.05, 0.05, 0.05, 1)

        Window.bind(on_resize=self._on_window_resize)

        def check_window_size(dt):
            w, h = Window.size
            print(f"[窗口] 当前尺寸: {w}x{h}")
            if w < h:
                self.root_layout.do_layout()
        Clock.schedule_once(check_window_size, 0.5)

        self.engine = AudioSpectrumEngine(
            samplerate=48000,
            blocksize=2048,
            channels=1
        )

        self.root_layout = BoxLayout(orientation='vertical')

        self.info_panel = InfoPanel()
        self.root_layout.add_widget(self.info_panel)

        content = BoxLayout(orientation='horizontal')

        self.spectrum_canvas = SpectrumCanvas()
        self.spectrum_canvas.samplerate = self.engine.samplerate
        content.add_widget(self.spectrum_canvas)

        self.control_panel = ControlPanel(self)
        content.add_widget(self.control_panel)

        self.root_layout.add_widget(content)

        self.update_event = None
        self.frame_count = 0
        self.last_fps_time = 0
        self.current_fps = 0

        return self.root_layout

    def start_capture(self):
        self.engine.stop_capture()

        self.spectrum_canvas.zoom_start = 0.0
        self.spectrum_canvas.zoom_end = 1.0
        self.spectrum_canvas.paused = False

        print("[调试] 开始启动音频采集...")

        self.engine.start_capture()

        def check_startup(dt):
            if not self.engine.is_running:
                self.info_panel.update_status('Audio Error')
                print("[调试] 音频采集启动失败!")
            else:
                self.info_panel.update_status('Capturing')
                print("[调试] 音频采集启动成功!")

                if self.update_event:
                    self.update_event.cancel()
                self.update_event = Clock.schedule_interval(self.update_display, 1.0 / 30.0)

        Clock.schedule_once(check_startup, 0.5)

    def stop_capture(self):
        self.engine.stop_capture()
        self.info_panel.update_status('Stopped')
        if self.update_event:
            self.update_event.cancel()
            self.update_event = None

    def pause_capture(self):
        self.engine.pause()
        self.spectrum_canvas.paused = True
        self.info_panel.update_status('Paused (scroll to zoom)')

    def resume_capture(self):
        self.engine.resume()
        self.spectrum_canvas.paused = False
        self.spectrum_canvas.zoom_start = 0.0
        self.spectrum_canvas.zoom_end = 1.0
        self.spectrum_canvas.draw()
        self.info_panel.update_status('Capturing')

    def update_display(self, dt):
        try:
            freqs, magnitude, time_data = self.engine.get_latest_data()

            if freqs is not None and magnitude is not None and len(freqs) > 0:
                peaks = self.engine.find_peaks(freqs, magnitude)
                self.spectrum_canvas.update_data(freqs, magnitude, time_data, peaks)
                self.info_panel.update_peaks(peaks)
            else:
                self.spectrum_canvas.draw()

            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
                self.info_panel.update_fps(self.current_fps)
        except Exception as e:
            print(f"[错误] update_display 异常: {e}")
            import traceback
            traceback.print_exc()

    def set_display_mode(self, time_domain):
        self.spectrum_canvas.show_time_domain = time_domain
        self.spectrum_canvas.draw()

    def set_fft_size(self, size):
        self.engine.fft_size = size

    def set_window_type(self, window_type):
        self.engine.window_type = window_type

    def set_log_scale(self, enabled):
        self.engine.log_scale = enabled
        self.spectrum_canvas.log_scale = enabled

    def set_dc_remove(self, enabled):
        self.engine.dc_remove = enabled

    def set_spectrum_avg(self, enabled):
        self.engine.spectrum_avg = enabled

    def set_peak_count(self, count):
        self.engine.peak_count = count

    def set_freq_min(self, freq_min):
        self.engine.freq_min = freq_min

    def set_freq_max(self, freq_max):
        self.engine.freq_max = freq_max

    def set_amp_max(self, amp_max):
        self.engine.amp_max = amp_max
        self.spectrum_canvas.amp_max = amp_max
        self.spectrum_canvas.draw()

    def _on_window_resize(self, window, width, height):
        print(f"[窗口] 尺寸变化: {width}x{height}")
        if hasattr(self, 'root_layout'):
            self.root_layout.do_layout()
        if hasattr(self, 'spectrum_canvas'):
            self.spectrum_canvas._update_canvas()

    def on_stop(self):
        self.stop_capture()
        self.spectrum_canvas.cleanup()
        Window.unbind(on_resize=self._on_window_resize)


def main():
    app = SpectrumAnalyzerApp()
    app.run()


if __name__ == '__main__':
    main()
