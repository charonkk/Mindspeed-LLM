# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的权重转换参数
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 8 \
       --add-qkv-bias \
       --load-dir ./model_from_hf/qwq_32b_hf/ \
       --save-dir ./model_weights/qwq_mcore/ \
       --tokenizer-model ./model_from_hf/qwq_32b_hf/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16 # --num-layer-list 11, 13, 19, 21 参数根据需要添加
