[app]
# 应用标题
title = Spectrum Analyzer
# 包名
package.name = spectrumanalyzer
# 包域名
package.domain = org.example
# 源代码目录
source.dir = .
# 主程序入口
source.include_exts = py,png,jpg,kv,atlas
# 版本号
version = 1.0.0
# 依赖项
requirements = python3,kivy==2.3.0,numpy,audiostream
# 图标（可选）
# icon.filename = %(source.dir)s/data/icon.png
# 权限
android.permissions = RECORD_AUDIO
# API级别
android.api = 33
android.minapi = 24
android.sdk = 33
android.ndk = 28c
# 架构（只打 arm64 减小体积和加快速度）
android.archs = arm64-v8a
# 屏幕方向
orientation = landscape
# 全屏
fullscreen = 0
# 日志级别
android.logcat_filters = *:S python:D
# Android 入口（包名.模块名）
android.entrypoint = org.kivy.android.PythonActivity
# ✅ 新增：接受 SDK 许可证
android.accept_sdk_license = True

[buildozer]
# Buildozer日志级别
log_level = 2
# 工作目录
build_dir = ./.buildozer
# 打包输出目录
bin_dir = ./bin
