#!/bin/bash

# 接收外部传入的参数
JSON_FILE=$1
BASE_URL=$2
MAX_CONCURRENT=4  # 最大并发下载数

# 参数校验
if [ -z "$JSON_FILE" ] || [ -z "$BASE_URL" ]; then
    echo "用法错误: $0 <json文件路径> <base_url>"
    echo "示例: $0 qwen3-coder-next.json https://modelscope.cn/models/.../resolve/master"
    exit 1
fi

if [ ! -f "$JSON_FILE" ]; then
    echo "错误: 找不到指定的 JSON 文件: $JSON_FILE"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "错误: 未找到 jq 命令，请先安装 jq (例如: apt-get install jq 或 yum install jq)。"
    exit 1
fi

# 根据传入的 json 文件名动态生成日志文件名，避免多个模型下载日志混在一起
LOG_FILE="download_${JSON_FILE%.*}.log"

verify_sha256() {
    local file=$1
    local expected_hash=$2
    
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    echo "正在校验: $file ..."
    local actual_hash=$(sha256sum "$file" | awk '{print $1}')
    
    if [ "$actual_hash" == "$expected_hash" ]; then
        echo "[成功] $file 校验通过"
        return 0
    else
        echo "[失败] $file 校验失败！预期: $expected_hash, 实际: $actual_hash"
        return 1
    fi
}

download_and_verify() {
    local name=$1
    local sha256=$2
    # 动态拼接 URL
    local url="${BASE_URL}/${name}"

    echo "开始下载: $name ..."
    
    wget --no-check-certificate -c "$url" -O "$name" -q

    if [ $? -eq 0 ]; then
        verify_sha256 "$name" "$sha256"
    else
        echo "[错误] $name 下载中断或失败"
    fi
}

# 导出环境变量和函数，供 xargs 中的子 shell 使用
export BASE_URL
export -f download_and_verify
export -f verify_sha256

echo "--- [$JSON_FILE] 下载任务开始于 $(date) ---" | tee -a "$LOG_FILE"

# 解析传入的 JSON 并发下载
jq -r '.Data.Files[] | .Name, .Sha256' "$JSON_FILE" | xargs -n 2 -P "$MAX_CONCURRENT" bash -c 'download_and_verify "$0" "$1"' | tee -a "$LOG_FILE"

echo "--- [$JSON_FILE] 下载任务结束于 $(date) ---" | tee -a "$LOG_FILE"
