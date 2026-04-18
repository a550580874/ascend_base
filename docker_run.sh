#!/bin/bash

# --- 配置部分 ---
CONTAINER_NAME="ming-test"
IMAGE_NAME="mindspeed_llm:rc4"  # 或者是你的镜像ID: ca1b56db4008
DATA_PATH="/home/"
NPU_DEVICES="0-7"

# --- 构造参数（这种方式最稳，不需要在行尾加反斜杠） ---
DOCKER_OPTS=""
DOCKER_OPTS+="-it -u root --ipc=host --net=host --privileged=true "
DOCKER_OPTS+="--name=${CONTAINER_NAME} "
DOCKER_OPTS+="-e ASCEND_VISIBLE_DEVICES=${NPU_DEVICES} "

# 挂载 NPU 设备
for i in {0..7}; do
    DOCKER_OPTS+="--device=/dev/davinci${i} "
done
DOCKER_OPTS+="--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc "

# 挂载 驱动、固件及数据
DOCKER_OPTS+="-v /etc/ascend_install.info:/etc/ascend_install.info "
DOCKER_OPTS+="-v /usr/local/Ascend/:/usr/local/Ascend/ "
DOCKER_OPTS+="-v /usr/local/Ascend/driver:/usr/local/Ascend/driver "
DOCKER_OPTS+="-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ "
DOCKER_OPTS+="-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi "
DOCKER_OPTS+="-v /usr/local/sbin/:/usr/local/sbin/ "
DOCKER_OPTS+="-v ${DATA_PATH}:${DATA_PATH} "

# 执行
echo "正在启动容器: ${CONTAINER_NAME} ..."
docker run ${DOCKER_OPTS} ${IMAGE_NAME} /bin/bash
