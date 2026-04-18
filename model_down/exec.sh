#!/bin/bash

# 1. 赋予核心下载脚本执行权限 (修复了你截图中名字不一致的问题)
chmod +x model_down.sh

# 2. 定义本次要下载的模型的变量
TARGET_JSON="qwen3-next-coder.json"
TARGET_URL="https://modelscope.cn/models/Qwen/Qwen3-Coder-Next/resolve/master"

# 3. 后台执行并传参
echo "准备在后台启动下载任务: $TARGET_JSON"
nohup ./model_down.sh "$TARGET_JSON" "$TARGET_URL" > download_terminal.log 2>&1 &

# 4. 打印提示信息
echo "任务已成功放入后台运行 (PID: $!)"
echo "你可以通过以下命令查看下载进度:"
echo "tail -f download_terminal.log"
