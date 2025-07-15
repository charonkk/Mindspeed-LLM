# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./ckpt/mamba2-tp8pp1 \
    --save-dir ./ckpt/mamba2-tp4pp1 \
    --target-pp-size 4 \
    --target-tp-size 1 \
    --d-model 2560 \
    --mamba-d-state 128 \
    --mamba2-head-dim 64 \
    --mamba2-n-groups 1 \

    # 注意，如果load权重不是训练后保存的权重，则需要增加如下配置参数
    # --input-tp-rank 4 \
    # --input-pp-rank 2 \
    # --num-layers 64 \
