#!/bin/bash

# =================配置区域=================
JSON_FILE="file.json"
BASE_URL="https://modelscope.cn/models/Qwen/Qwen3-Coder-Next/resolve/master"
MAX_CONCURRENT=4  # 最大并发下载数
LOG_FILE="download.log"
# ==========================================

# 检查依赖
if ! command -v jq &> /dev/null; then
    echo "错误: 未找到 jq 命令，请先安装 jq。"
    exit 1
fi

# 校验函数
verify_sha256() {
    local file=$1
    local expected_hash=$2
    
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    echo "正在校验: $file ..."
    # 获取计算出的 hash (适配 Linux 环境下的 sha256sum)
    actual_hash=$(sha256sum "$file" | awk '{print $1}')
    
    if [ "$actual_hash" == "$expected_hash" ]; then
        echo "[成功] $file 校验通过"
        return 0
    else
        echo "[失败] $file 校验失败！预期: $expected_hash, 实际: $actual_hash"
        return 1
    fi
}

# 单个文件下载与校验逻辑
download_and_verify() {
    local name=$1
    local sha256=$2
    local url="${BASE_URL}/${name}"

    echo "开始下载: $name ..."
    
    # wget 参数说明:
    # -c: 断点续传
    # --no-check-certificate: 忽略证书错误 (你要求的)
    # -q: 静默模式，避免日志过大（若需看进度可删掉）
    # -O: 指定文件名
    wget --no-check-certificate -c "$url" -O "$name" -q

    if [ $? -eq 0 ]; then
        verify_sha256 "$name" "$sha256"
    else
        echo "[错误] $name 下载中断或失败"
    fi
}

export -f download_and_verify
export -f verify_sha256
export BASE_URL

echo "--- 任务开始于 $(date) ---" | tee -a $LOG_FILE

# 从 JSON 提取文件名和 SHA256，利用 xargs 实现并发控制
# -P: 控制并行数
# -n 2: 每次传递两个参数给函数 (Name 和 Sha256)
jq -r '.Data.Files[] | .Name, .Sha256' "$JSON_FILE" | xargs -n 2 -P "$MAX_CONCURRENT" bash -c 'download_and_verify "$0" "$1"' | tee -a $LOG_FILE

echo "--- 任务结束于 $(date) ---" | tee -a $LOG_FILE
