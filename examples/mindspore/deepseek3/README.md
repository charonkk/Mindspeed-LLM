## DeepSeek-V3 指南
### 1. 准备工作

参考[安装指导](../../../docs/mindspore/features/install_guide.md)，完成环境安装和[预训练数据处理](../../../docs/pytorch/solutions/pretrain/pretrain_dataset.md)。


### 2. 权重转换
MindSpore后端已支持DeepSeek3模型权重转换，使用方式与PyTorch后端一致，详细转换参考[DeepSeek3权重转换](../../mcore/deepseek3/README.md)。

**注意：**
- 当前尚不支持QLoRA权重量化转换，【--qlora-nf4】参数仅可置为False。
- MindSpore 后端默认在Device侧进行权重转换，在模型较大时存在OOM风险，因此建议用户手动修改`examples/mcore/deepseek3/convert_ckpt_deepseek3.py`，在包导入时加入如下代码设置CPU侧执行权重转换：

```python
import mindspore as ms
ms.set_context(device_target="CPU", pynative_synchronize=True)
import torch
torch.configs.set_pyboost(False)
```

- MindSpore 后端转换出的模型权重无法用于 Torch后端训练或推理。

### 3. 预训练

#### 参数配置
预训练脚本需根据实际情况修改参数配置，包括路径配置、并行配置、模型参数配置等。

路径配置包括权重保存路径、权重加载路径、词表路径、数据集路径，示例如下：

``` shell
# 根据实际情况配置权重保存、权重加载、词表、数据集路径
# 注意：提供的路径需要加双引号
CKPT_SAVE_DIR="./ckpt/deepseek3"  # 训练完成后的权重保存路径
CKPT_LOAD_DIR="./model_weights/deepseek3/"  # 权重加载路径，填入权重转换时保存的权重路径
DATA_PATH="./dataset/enwiki_text_document"  # 数据集路径，填入数据预处理时保存的数据路径
TOKENIZER_MODEL="./model_from_hf/deepseek3/tokenizer"  # 词表路径，填入下载的开源权重词表路径
```

【单机运行】
相关配置参数需要修改为单机配置，供参考的单机配置修改如下：
```shell
...
NNODES=1
NODE_RANK=0  
...
TP=1
PP=2
EP=4
...
NUM_LAYERS=4
...
GBS=8
...
MOE_ARGS="
    ...
    --first-k-dense-replace 1 \
    ...
    --num-experts 16 \
    ...
"
...
GPT_ARGS="
    ...
    --noop-layers 0 \
    ...
"
...
```
【多机运行】
```shell
# 根据分布式集群实际情况配置分布式参数
GPUS_PER_NODE=8
MASTER_ADDR="your master node IP" #主节点IP
MASTER_PORT=6000
NNODES=4 # 集群里的节点数，以实际情况填写
NODE_RANK="current node rank" # 当前节点RANK,主节点为0，其他可以使1,2..
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
```

#### 启动预训练

```shell
# 初始化环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
# 注意：在MindSpeed-LLM目录下执行
sh examples/mindspore/deepseek3/pretrain_deepseek3_671B_4k_ms.sh
```



**注意：**

- 多机训练需在多个终端同时启动预训练脚本。
- 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加--no-shared-storage参数，设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。
- 小网场景下同时使能dualpipe和MoE零冗余通信(--moe-zerc)特性时，性能波动较大且收益不明显，不建议开启。
- 当前--reuse-fp32-param需跟--use-distributed-optimizer参数一起使用，暂不支持单独使用。