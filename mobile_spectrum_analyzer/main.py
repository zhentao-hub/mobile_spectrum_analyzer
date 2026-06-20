"""
Mobile Spectrum Analyzer - Kivy Version
Supports real-time microphone capture + FFT spectrum display
"""
import os
import sys
import numpy as np

# Kivy 配置
os.environ['KIVY_NO_CONSOLELOG'] = '0'  # 显示控制台日志，便于调试
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
        self.show_time_domain = False  # False=频域, True=时域
        self.log_scale = False
        self.amp_max = 0.3  # Y轴最大幅值显示范围
        self.paused = False  # 是否暂停
        
        # 缩放参数（仅暂停时有效）
        self.zoom_start = 0.0  # 显示频率范围起点（相对0-1）
        self.zoom_end = 1.0    # 显示频率范围终点（相对0-1）
        
        # 拖动平移参数
        self.dragging = False
        self.last_touch_x = 0

        # 颜色配置
        self.grid_color = (0.3, 0.3, 0.3, 1)
        self.spectrum_color = (0, 1, 1, 1)
        self.peak_color = (1, 0, 0, 1)
        self.time_color = (1, 1, 0, 1)
        self.text_color = (1, 1, 1, 1)  # 白色，更醒目

        # 网格线
        self.grid_lines = []
        self.spectrum_line = None
        self.time_line = None
        self.peak_labels = []
        
        # 坐标轴标签
        self.axis_labels = []
        self.axis_titles = []
        
        # 峰值文字标签
        self.peak_text_labels = []
        
        # 十字光标和信息标签
        self.crosshair_line = None
        self.cursor_info_label = None

        with self.canvas:
            # 背景
            Color(0.1, 0.1, 0.1, 1)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)

            # 网格
            self.grid_color_inst = Color(*self.grid_color)
            for _ in range(20):
                self.grid_lines.append(Line(points=[], width=1))

            # 频谱线
            self.spectrum_color_inst = Color(*self.spectrum_color)
            self.spectrum_line = Line(points=[], width=2)

            # 时域线
            self.time_color_inst = Color(*self.time_color)
            self.time_line = Line(points=[], width=1.5)

            # 峰值标记
            self.peak_color_inst = Color(*self.peak_color)
            self.peak_ellipses = []
            self.peak_texts = []
            
            # 十字光标（初始隐藏）
            Color(1, 1, 0, 0.6)
            self.crosshair_line = Line(points=[], width=1)

        # 坐标轴标签容器
        self.label_layout = FloatLayout(pos=self.pos, size=self.size)
        self.add_widget(self.label_layout)
        
        # 峰值标签专用容器（单独清空，避免残留）
        self.peak_label_layout = FloatLayout(pos=self.pos, size=self.size)
        self.add_widget(self.peak_label_layout)
        
        # 光标信息标签（显示频率和幅值）
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
        
        # 创建坐标轴标签和标题
        self._create_axis_labels()

        self.bind(pos=self._update_canvas, size=self._update_canvas)
        self.bind(pos=self._update_peak_layout, size=self._update_peak_layout)
        
        # 绑定全局鼠标位置事件，支持纯悬停显示光标
        Window.bind(mouse_pos=self._on_mouse_pos)
    
    def _on_mouse_pos(self, window, pos):
        """处理全局鼠标移动，实现悬停遍历"""
        # 转换窗口坐标到本widget的局部坐标
        touch_x, touch_y = self.to_widget(*pos)
        
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()
        
        # 检查是否在绘图区域内
        in_plot = (plot_x <= touch_x <= plot_x + plot_w and plot_y <= touch_y <= plot_y + plot_h)
        
        if not in_plot or self.show_time_domain:
            self._hide_cursor()
            return
        
        # 频域模式下显示十字光标和信息
        self._update_cursor(touch_x, touch_y)
    
    def _update_peak_layout(self, *args):
        """更新峰值标签布局位置和大小"""
        self.peak_label_layout.pos = self.pos
        self.peak_label_layout.size = self.size
    
    def _create_axis_labels(self):
        """创建坐标轴刻度标签"""
        # 清除旧标签
        for label in self.axis_labels + self.axis_titles:
            if label in self.label_layout.children:
                self.label_layout.remove_widget(label)
        self.axis_labels = []
        self.axis_titles = []
        
        # X轴标题
        x_title = Label(
            text='Frequency (Hz)',
            font_size='12sp',
            color=(0.5, 0.8, 1, 1),
            size_hint=(None, None),
            size=(120, 20)
        )
        self.axis_titles.append(x_title)
        self.label_layout.add_widget(x_title)
        
        # Y轴标题
        y_title = Label(
            text='Magnitude',
            font_size='12sp',
            color=(0.5, 0.8, 1, 1),
            size_hint=(None, None),
            size=(100, 20)
        )
        self.axis_titles.append(y_title)
        self.label_layout.add_widget(y_title)
        
        # X轴标签 (频率)
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
        
        # Y轴标签 (幅值)
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
        """处理鼠标按下：滚轮缩放或开始拖动平移"""
        if not self.paused or self.show_time_domain:
            return super().on_touch_down(touch)

        # 将窗口坐标转换到本 widget 局部坐标
        touch_x, touch_y = self.to_widget(*touch.pos)

        # 检查触摸是否在绘图区域内
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()
        if not (plot_x <= touch_x <= plot_x + plot_w and plot_y <= touch_y <= plot_y + plot_h):
            return super().on_touch_down(touch)

        # 检测滚轮方向
        if touch.button == 'scrollup':
            # 放大（缩小显示范围）
            zoom_factor = 0.9
        elif touch.button == 'scrolldown':
            # 缩小（放大显示范围）
            zoom_factor = 1.1
        elif touch.button == 'left':
            # 开始拖动平移
            self.dragging = True
            self.last_touch_x = touch_x
            return True
        else:
            return super().on_touch_down(touch)

        # 计算鼠标在绘图区域的水平位置（0-1）
        mouse_ratio = (touch_x - plot_x) / plot_w if plot_w > 0 else 0.5

        # 当前显示范围
        current_range = self.zoom_end - self.zoom_start
        new_range = max(0.05, min(1.0, current_range * zoom_factor))

        # 以鼠标位置为中心缩放
        center = self.zoom_start + current_range * mouse_ratio
        new_start = max(0.0, center - new_range * mouse_ratio)
        new_end = min(1.0, new_start + new_range)

        # 调整 start 如果 end 超出范围
        if new_end >= 1.0:
            new_end = 1.0
            new_start = max(0.0, new_end - new_range)

        self.zoom_start = new_start
        self.zoom_end = new_end
        self.draw()
        return True
    
    def on_touch_move(self, touch):
        """处理鼠标移动：拖动平移或显示十字光标"""
        touch_x, touch_y = self.to_widget(*touch.pos)
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()

        # 检查是否在绘图区域内
        in_plot = (plot_x <= touch_x <= plot_x + plot_w and plot_y <= touch_y <= plot_y + plot_h)

        if not in_plot:
            self._hide_cursor()
            if self.dragging:
                self.dragging = False
            return super().on_touch_move(touch)

        # 拖动平移（暂停且非时域模式）
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

        # 频域模式下显示十字光标和信息
        if not self.show_time_domain:
            self._update_cursor(touch_x, touch_y)
            return True

        return super().on_touch_move(touch)
    
    def _update_cursor(self, touch_x, touch_y):
        """更新十字光标和信息标签"""
        if self.freqs is None or self.magnitude is None or len(self.freqs) == 0:
            self._hide_cursor()
            return
        
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()
        if plot_w <= 0:
            self._hide_cursor()
            return
        
        # 计算鼠标对应的频率
        mouse_ratio = (touch_x - plot_x) / plot_w
        f_min_all, f_max_all = self.freqs[0], self.freqs[-1]
        freq_range_all = f_max_all - f_min_all if f_max_all > f_min_all else 1
        
        # 考虑缩放
        if self.paused:
            f_min = f_min_all + freq_range_all * self.zoom_start
            f_max = f_min_all + freq_range_all * self.zoom_end
        else:
            f_min = f_min_all
            f_max = f_max_all
        
        freq = f_min + (f_max - f_min) * mouse_ratio
        
        # 找到最近的频率点
        idx = np.argmin(np.abs(self.freqs - freq))
        actual_freq = self.freqs[idx]
        actual_mag = self.magnitude[idx]
        
        # 峰值吸附：如果鼠标靠近某个峰值，就吸附到该峰值
        is_peak = False
        if self.peaks and len(self.peaks) > 0:
            # 找到最近的峰值
            nearest_peak = min(self.peaks, key=lambda p: abs(float(p[0]) - freq))
            peak_freq = float(nearest_peak[0])
            peak_mag = float(nearest_peak[1])
            
            # 吸附阈值：显示频率范围的3%，最小20Hz
            threshold = max((f_max - f_min) * 0.03, 20)
            
            if abs(peak_freq - freq) < threshold:
                actual_freq = peak_freq
                actual_mag = peak_mag
                is_peak = True
                # 重新计算索引用于显示
                idx = np.argmin(np.abs(self.freqs - peak_freq))
        
        # 计算该点在屏幕上的x坐标
        px = plot_x + plot_w * (actual_freq - f_min) / (f_max - f_min) if (f_max > f_min) else plot_x
        
        # 更新十字线
        self.crosshair_line.points = [float(px), float(plot_y), float(px), float(plot_y + plot_h)]
        
        # 更新信息标签
        if self.log_scale:
            mag_text = f'{actual_mag:.1f} dB'
        else:
            mag_text = f'{actual_mag:.4f}'
        
        peak_marker = ' [color=ffaa00](Peak)[/color]' if is_peak else ''
        self.cursor_info_label.text = f'[b]{actual_freq:.0f} Hz{peak_marker}\n{mag_text}[/b]'
        self.cursor_info_label.pos = (float(px + 10), float(touch_y - 10))
    
    def _hide_cursor(self):
        """隐藏十字光标和信息标签"""
        self.crosshair_line.points = []
        self.cursor_info_label.text = ''
    
    def on_touch_up(self, touch):
        """处理鼠标释放，结束拖动"""
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
        
        # 留出边距给坐标轴标签
        left_margin = dp(60)
        bottom_margin = dp(30)
        right_margin = dp(10)
        top_margin = dp(10)
        
        plot_x = x + left_margin
        plot_y = y + bottom_margin
        plot_w = w - left_margin - right_margin
        plot_h = h - bottom_margin - top_margin

        # 水平网格线 (5条)
        for i, line in enumerate(self.grid_lines[:5]):
            gy = plot_y + plot_h * (i + 1) / 6
            line.points = [plot_x, gy, plot_x + plot_w, gy]

        # 垂直网格线 (5条)
        for i, line in enumerate(self.grid_lines[5:10]):
            gx = plot_x + plot_w * (i + 1) / 6
            line.points = [gx, plot_y, gx, plot_y + plot_h]

        # 隐藏多余的线
        for line in self.grid_lines[10:]:
            line.points = []

    def _get_plot_area(self):
        """获取实际绘图区域（留出坐标轴标签空间）"""
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
            self._update_axis_labels(0, 20000, 0, 1)
            return

        # 获取绘图区域
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()

        # 频率范围
        f_min_all, f_max_all = self.freqs[0], self.freqs[-1]
        freq_range_all = f_max_all - f_min_all if f_max_all > f_min_all else 1
        
        # 应用缩放（仅暂停时）
        if self.paused:
            f_min = f_min_all + freq_range_all * self.zoom_start
            f_max = f_min_all + freq_range_all * self.zoom_end
        else:
            f_min = f_min_all
            f_max = f_max_all
        
        # 裁剪数据到显示范围
        mask = (self.freqs >= f_min) & (self.freqs <= f_max)
        display_freqs = self.freqs[mask]
        display_mags = self.magnitude[mask]
        
        if len(display_freqs) == 0:
            self.spectrum_line.points = []
            self._clear_peak_labels()
            self._update_axis_labels(f_min, f_max, 0, 1)
            return
        
        freq_range = f_max - f_min if f_max > f_min else 1

        # 幅值范围
        if self.log_scale:
            mag_min = max(-120, np.min(display_mags) - 10)
            mag_max = max(0, np.max(display_mags) + 10)
        else:
            mag_min = 0
            # 使用用户设置的 amp_max 作为固定的Y轴最大值
            mag_max = max(self.amp_max, 0.001)

        mag_range = mag_max - mag_min if mag_max > mag_min else 1

        # 生成频谱线点
        points = []
        n = len(display_freqs)
        max_points = min(n, 800)  # 限制点数保证性能
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

        # 更新坐标轴标签
        self._update_axis_labels(f_min, f_max, mag_min, mag_max)

        # 绘制峰值（只在显示范围内）
        display_peaks = [(f, m) for f, m in self.peaks if f_min <= f <= f_max]
        self._draw_peaks(f_min, freq_range, mag_min, mag_range, plot_x, plot_y, plot_w, plot_h, display_peaks)
    
    def _update_axis_labels(self, f_min, f_max, mag_min, mag_max):
        """更新坐标轴刻度标签"""
        x, y = self.pos
        w, h = self.size
        
        # 留出边距给Y轴标签
        left_margin = dp(60)
        bottom_margin = dp(30)
        right_margin = dp(10)
        top_margin = dp(10)
        
        plot_x = x + left_margin
        plot_y = y + bottom_margin
        plot_w = w - left_margin - right_margin
        plot_h = h - bottom_margin - top_margin
        
        # 更新标题位置（转换为Python float）
        if self.axis_titles:
            self.axis_titles[0].pos = (float(plot_x + plot_w / 2 - 60), float(y - 5))  # X轴标题
            self.axis_titles[1].pos = (float(x - 5), float(plot_y + plot_h / 2 - 10))  # Y轴标题
        
        # 更新X轴标签 (频率)
        for i in range(6):
            label = self.axis_labels[i]
            freq_val = float(f_min + (f_max - f_min) * i / 5)
            if freq_val >= 1000:
                label.text = f'{freq_val/1000:.1f}k'
            else:
                label.text = f'{freq_val:.0f}'
            label.pos = (float(plot_x + plot_w * i / 5 - 40), float(y + 5))
        
        # 更新Y轴标签 (幅值)
        for i in range(6):
            label = self.axis_labels[6 + i]
            mag_val = float(mag_min + (mag_max - mag_min) * i / 5)
            if self.log_scale:
                label.text = f'{mag_val:.0f}dB'
            else:
                if mag_max >= 1:
                    label.text = f'{mag_val:.2f}'
                elif mag_max >= 0.1:
                    label.text = f'{mag_val:.3f}'
                else:
                    label.text = f'{mag_val:.4f}'
            label.pos = (float(x + 5), float(plot_y + plot_h * i / 5 - 10))

    def _draw_time_domain(self):
        # 时域模式下不显示峰值标注
        self._clear_peak_labels()
        
        if self.time_data is None or len(self.time_data) == 0:
            self.time_line.points = []
            return

        # 获取绘图区域
        plot_x, plot_y, plot_w, plot_h = self._get_plot_area()

        data = self.time_data
        n = len(data)
        max_points = min(n, 800)
        step = max(1, n // max_points)

        # 归一化
        data_max = np.max(np.abs(data))
        if data_max > 0:
            data = data / data_max

        points = []
        for i in range(0, n, step):
            tx = plot_x + plot_w * i / n
            ty = plot_y + plot_h * 0.5 + plot_h * 0.45 * data[i]
            points.extend([tx, ty])

        self.time_line.points = points
        self.spectrum_line.points = []

    def _clear_peak_labels(self):
        """彻底清除所有峰值相关的图形和标签"""
        # 清除峰值圆点
        for ellipse in self.peak_ellipses:
            if ellipse in self.canvas.children:
                self.canvas.remove(ellipse)
        self.peak_ellipses = []
        
        # 清除峰值文字标签（保留光标信息标签）
        for widget in list(self.peak_label_layout.children):
            if widget is not self.cursor_info_label:
                self.peak_label_layout.remove_widget(widget)
        self.peak_text_labels = []
        
        # 清除连接线
        for line in getattr(self, 'peak_lines', []):
            if line in self.canvas.children:
                self.canvas.remove(line)
        self.peak_lines = []

    def _draw_peaks(self, f_min, freq_range, mag_min, mag_range, plot_x=None, plot_y=None, plot_w=None, plot_h=None, peaks=None):
        # 先彻底清除旧的峰值标注
        self._clear_peak_labels()

        if peaks is None:
            peaks = self.peaks
        
        if not peaks:
            return

        if plot_x is None:
            plot_x, plot_y, plot_w, plot_h = self._get_plot_area()

        # 只显示最强的前3个峰值，避免标签重叠
        sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:3]
        
        # 计算每个峰值的位置
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
            # 绘制峰值圆点
            Color(*self.peak_color)
            for px, py, freq, mag in peak_positions:
                ellipse = Ellipse(pos=(float(px - 4), float(py - 4)), size=(8, 8))
                self.peak_ellipses.append(ellipse)
        
        # 绘制峰值文字标签（直接放在峰值点旁边，不画连线）
        for i, (px, py, freq, mag) in enumerate(peak_positions):
            # 直接放在峰值点右上方，如果多个峰值太近则稍微错开
            label_x = px + dp(8)
            label_y = py + dp(8) + i * dp(16)
            
            # 确保不超出绘图区域顶部
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
        self.width = dp(280)  # 控制面板宽度
        self.bar_width = dp(10)  # 滚动条宽度
        self.scroll_type = ['bars', 'content']
        
        # 内部容器
        self.layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(900),  # 总高度，根据内容调整
            padding=dp(10),
            spacing=dp(8)
        )
        self.layout.bind(minimum_height=self.layout.setter('height'))
        self.add_widget(self.layout)

        # Title
        self.layout.add_widget(Label(
            text='[b]Spectrum Control[/b]',
            markup=True,
            size_hint_y=None,
            height=dp(30),
            color=(1, 1, 1, 1)
        ))

        # Start/Stop buttons
        btn_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(60), spacing=dp(10))
        self.start_btn = Button(
            text='▶ Start',
            on_press=self.on_start,
            font_size='18sp',
            background_color=(0.2, 0.7, 0.2, 1),  # 绿色背景
            color=(1, 1, 1, 1),  # 白色文字
            size_hint=(1, None),
            height=dp(50)
        )
        self.stop_btn = Button(
            text='⏹ Stop',
            on_press=self.on_stop,
            disabled=True,
            font_size='18sp',
            background_color=(0.7, 0.2, 0.2, 1),  # 红色背景
            color=(1, 1, 1, 1),  # 白色文字
            size_hint=(1, None),
            height=dp(50)
        )
        btn_layout.add_widget(self.start_btn)
        btn_layout.add_widget(self.stop_btn)
        self.layout.add_widget(btn_layout)

        # Pause button
        self.pause_btn = ToggleButton(
            text='⏸ Pause',
            size_hint_y=None,
            height=dp(50),
            on_press=self.on_pause,
            disabled=True,  # 初始禁用，开始采集后启用
            font_size='18sp',
            background_color=(0.7, 0.5, 0.2, 1),  # 橙色背景
            color=(1, 1, 1, 1),  # 白色文字
        )
        self.layout.add_widget(self.pause_btn)

        # Display mode toggle
        self.layout.add_widget(Label(text='Display Mode:', size_hint_y=None, height=dp(25), color=(0.8, 0.8, 0.8, 1)))
        mode_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        self.freq_mode_btn = ToggleButton(text='Frequency', group='mode', state='down')
        self.time_mode_btn = ToggleButton(text='Time', group='mode')
        self.freq_mode_btn.bind(on_press=lambda x: self.on_mode_change(False))
        self.time_mode_btn.bind(on_press=lambda x: self.on_mode_change(True))
        mode_layout.add_widget(self.freq_mode_btn)
        mode_layout.add_widget(self.time_mode_btn)
        self.layout.add_widget(mode_layout)

        # FFT Size
        self.layout.add_widget(Label(text='FFT Size:', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        self.fft_slider = Slider(min=8, max=14, value=11, step=1, size_hint_y=None, height=dp(40))
        self.fft_label = Label(text='2048', size_hint_y=None, height=dp(20), color=(1, 1, 1, 1))
        self.fft_slider.bind(value=self.on_fft_change)
        self.layout.add_widget(self.fft_slider)
        self.layout.add_widget(self.fft_label)

        # Window function
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

        # Log scale
        self.log_btn = ToggleButton(
            text='Log Scale (dB)',
            size_hint_y=None,
            height=dp(40)
        )
        self.log_btn.bind(on_press=self.on_log_change)
        self.layout.add_widget(self.log_btn)

        # DC removal
        self.dc_btn = ToggleButton(
            text='Remove DC',
            state='down',
            size_hint_y=None,
            height=dp(40)
        )
        self.dc_btn.bind(on_press=self.on_dc_change)
        self.layout.add_widget(self.dc_btn)

        # Spectrum averaging
        self.avg_btn = ToggleButton(
            text='Spectrum Avg',
            size_hint_y=None,
            height=dp(40)
        )
        self.avg_btn.bind(on_press=self.on_avg_change)
        self.layout.add_widget(self.avg_btn)

        # Peak count
        self.layout.add_widget(Label(text='Peak Count:', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        self.peak_slider = Slider(min=0, max=5, value=3, step=1, size_hint_y=None, height=dp(40))
        self.peak_label = Label(text='3', size_hint_y=None, height=dp(20), color=(1, 1, 1, 1))
        self.peak_slider.bind(value=self.on_peak_change)
        self.layout.add_widget(self.peak_slider)
        self.layout.add_widget(self.peak_label)

        # Frequency range
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
        
        # Magnitude range
        self.layout.add_widget(Label(text='Max Magnitude:', size_hint_y=None, height=dp(20), color=(0.8, 0.8, 0.8, 1)))
        self.amp_max_slider = Slider(min=0.001, max=0.3, value=0.3, step=0.001, size_hint_y=None, height=dp(40))
        self.amp_max_label = Label(text='0.300', size_hint_y=None, height=dp(20), color=(1, 1, 1, 1))
        self.amp_max_slider.bind(value=self.on_amp_max_change)
        self.layout.add_widget(self.amp_max_slider)
        self.layout.add_widget(self.amp_max_label)

        # Spacer
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
        # Set window size for desktop only
        if sys.platform in ('win32', 'darwin', 'linux'):
            Window.size = (1200, 800)
        Window.clearcolor = (0.05, 0.05, 0.05, 1)

        # === Android 横屏全屏修复 ===
        # 绑定窗口尺寸变化事件，强制更新布局消除黑边
        Window.bind(on_resize=self._on_window_resize)

        # 延迟检查窗口尺寸，确保横屏后正确适应
        def check_window_size(dt):
            w, h = Window.size
            print(f"[窗口] 当前尺寸: {w}x{h}")
            if w < h:  # 如果是竖屏比例，强制重新布局
                self.root_layout.do_layout()
        Clock.schedule_once(check_window_size, 0.5)
        # ============================


        # 创建音频引擎
        self.engine = AudioSpectrumEngine(
            samplerate=48000,
            blocksize=2048,
            channels=1
        )

        # 主布局
        self.root_layout = BoxLayout(orientation='vertical')

        # 顶部信息栏
        self.info_panel = InfoPanel()
        self.root_layout.add_widget(self.info_panel)

        # 内容区域
        content = BoxLayout(orientation='horizontal')

        # 频谱画布
        self.spectrum_canvas = SpectrumCanvas()
        content.add_widget(self.spectrum_canvas)

        # 控制面板
        self.control_panel = ControlPanel(self)
        content.add_widget(self.control_panel)

        self.root_layout.add_widget(content)

        # 定时更新
        self.update_event = None
        self.frame_count = 0
        self.last_fps_time = 0
        self.current_fps = 0

        return self.root_layout

    def start_capture(self):
        """Start capture"""
        # 确保之前的流已停止
        self.engine.stop_capture()
        
        # 重置缩放
        self.spectrum_canvas.zoom_start = 0.0
        self.spectrum_canvas.zoom_end = 1.0
        self.spectrum_canvas.paused = False
        
        print("[调试] 开始启动音频采集...")
        
        # 启动音频采集
        self.engine.start_capture()
        
        # 延迟检查是否成功启动
        from kivy.clock import Clock
        def check_startup(dt):
            if not self.engine.is_running:
                self.info_panel.update_status('Audio Error')
                print("[调试] 音频采集启动失败!")
            else:
                self.info_panel.update_status('Capturing')
                print("[调试] 音频采集启动成功!")
                
                # 启动显示更新
                if self.update_event:
                    self.update_event.cancel()
                self.update_event = Clock.schedule_interval(self.update_display, 1.0 / 30.0)
        
        Clock.schedule_once(check_startup, 0.5)

    def stop_capture(self):
        """Stop capture"""
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
        """Update display"""
        try:
            freqs, magnitude, time_data = self.engine.get_latest_data()

            if freqs is not None and magnitude is not None and len(freqs) > 0:
                peaks = self.engine.find_peaks(freqs, magnitude)
                self.spectrum_canvas.update_data(freqs, magnitude, time_data, peaks)
                self.info_panel.update_peaks(peaks)
            else:
                # 显示提示
                self.spectrum_canvas.draw()

            # 计算FPS
            self.frame_count += 1
            import time
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
        self.spectrum_canvas.amp_max = amp_max
        self.spectrum_canvas.draw()
        self.spectrum_canvas.amp_max = amp_max
        self.spectrum_canvas.draw()
        self.spectrum_canvas.amp_max = amp_max
        self.spectrum_canvas.draw()
        self.spectrum_canvas.amp_max = amp_max
        self.spectrum_canvas.draw()
        self.spectrum_canvas.amp_max = amp_max
        self.spectrum_canvas.draw()
        self.spectrum_canvas.amp_max = amp_max
        self.spectrum_canvas.draw()


    def _on_window_resize(self, window, width, height):
        """窗口尺寸变化时强制更新布局，修复横屏黑边问题"""
        print(f"[窗口] 尺寸变化: {width}x{height}")
        # 强制所有布局重新计算
        if hasattr(self, 'root_layout'):
            self.root_layout.do_layout()
        # 强制频谱画布重绘
        if hasattr(self, 'spectrum_canvas'):
            self.spectrum_canvas._update_canvas()

    def on_stop(self):
        self.stop_capture()


def main():
    app = SpectrumAnalyzerApp()
    app.run()


if __name__ == '__main__':
    main()
