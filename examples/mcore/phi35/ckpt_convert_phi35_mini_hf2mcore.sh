# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type-hf phi3.5 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --target-tensor-parallel-size 1 \
   --target-pipeline-parallel-size 8 \
   --load-dir ./model_from_hf/Phi-3.5-mini-instruct/ \
   --save-dir ./model_weights/phi35_mini_mcore/ \
   --tokenizer-model ./model_from_hf/Phi-3.5-mini-instruct/tokenizer.model \
   #  --num-layers-per-virtual-pipeline-stage 1  当转换pretain使用的权重时，增加该参数
