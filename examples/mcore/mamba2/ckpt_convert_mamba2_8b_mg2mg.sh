# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py \
    --load-model-type mg \
    --save-model-type mg \
    --load-dir ./ckpt/mamba2-tp4pp2 \
    --save-dir ./ckpt/mamba2-tp2pp4 \
    --target-pp-size 4 \
    --target-tp-size 2 \

    # 注意，如果load权重不是训练后保存的权重，则需要增加如下配置参数
    # --input-tp-rank 4 \
    # --input-pp-rank 2 \
