# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

# 权重格式转换
# 使用内存至少2T的主机来转换本权重
python convert_ckpt.py \
   --use-mcore-models \
   --moe-grouped-gemm \
   --model-type-hf deepseek2 \
   --model-type GPT \
   --load-model-type mg \
   --save-model-type hf \
   --params-dtype bf16 \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 1 \
   --target-expert-parallel-size 1 \
   --load-dir ./model_weights/deepseek25-mcore/ \
   --save-dir ./model_from_hf/deepseek25-hf/ \
   --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec
