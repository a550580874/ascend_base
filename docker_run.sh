docker run -it -u root --ipc=host --net=host  --name=ming-test --ipc=host  --privileged=true \ # name参数代表新建容器的名字
-e ASCEND_VISIBLE_DEVICES=0-7 \ # 新建容器调用八卡
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/ascend_install.info:/etc/ascend_install.info \ # 必须挂载
-v /usr/local/Ascend/:/usr/local/Ascend/ \ # 挂载宿主机上CANN驱动和固件
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \  # 挂载npu-smi命令
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /home/:/home/ \ # 挂载模型数据文件
mindspeed_llm:rc4 \ # 基础镜像名字和版本
/bin/bash
