# 移动端频谱分析器

基于 Kivy 框架的跨平台实时音频频谱分析器，支持 Android 手机麦克风采集。

## 功能特性

- **实时麦克风采集**：使用手机麦克风实时采集音频
- **FFT 频谱分析**：支持多种 FFT 参数配置
- **时域/频域切换**：实时切换波形显示和频谱显示
- **峰值检测**：自动标注频谱峰值频率
- **多种窗函数**：Hann / Hamming / Blackman / Rectangular
- **对数刻度**：支持 dB 刻度显示
- **滑动平均**：频谱平滑处理
- **频率范围调节**：可调节显示频率上限

## 文件结构

```
mobile_spectrum_analyzer/
├── main.py              # 主程序（Kivy UI）
├── audio_engine.py      # 音频采集与 FFT 引擎
├── buildozer.spec       # Android 打包配置
└── README.md            # 说明文档
```

## 桌面端运行

### 安装依赖

```bash
pip install kivy numpy scipy pyaudio
```

### 运行

```bash
python main.py
```

## Android 打包

### 1. 安装 Buildozer

```bash
pip install buildozer
```

### 2. 安装 Android 依赖

```bash
sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev
```

### 3. 打包 APK

```bash
cd mobile_spectrum_analyzer
buildozer android debug
```

打包完成后，APK 文件位于 `./bin/` 目录。

### 4. 部署到手机

```bash
buildozer android debug deploy run
```

## 使用说明

1. 点击 **开始** 按钮启动麦克风采集
2. 频谱图实时显示麦克风采集到的音频频谱
3. 点击 **时域** 切换到时域波形显示
4. 调节 **FFT Size** 改变频率分辨率
5. 选择 **窗函数** 优化频谱泄漏
6. 开启 **对数刻度** 以 dB 显示幅值
7. 调节 **峰值标注数量** 显示主要频率成分

## 技术栈

| 组件 | 说明 |
|------|------|
| Kivy | 跨平台 UI 框架 |
| audiostream | 移动端音频采集 |
| numpy | FFT 计算 |
| scipy | 峰值检测 |
| PyAudio | 桌面端音频采集（回退） |

## 注意事项

- Android 需要授予 **录音权限**
- 建议横屏使用以获得更好的显示效果
- 采样率固定为 48kHz
- FFT Size 越大，频率分辨率越高，但刷新率会降低
