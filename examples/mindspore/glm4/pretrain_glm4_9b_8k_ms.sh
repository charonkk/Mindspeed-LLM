#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=2
PP=2

DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    --join False
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-mcore-models \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --num-layers 40 \
    --hidden-size 4096 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --max-position-embeddings 8192 \
    --padded-vocab-size 151552 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 2 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --use-glm-rope \
    --rotary-percent 0.5 \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --lr 1.25e-6 \
    --norm-epsilon 1.5625e-07 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096 \
    --adam-beta2 0.95 \
    --no-load-optim \
    --no-load-rng \
    --no-gradient-accumulation-fusion \
    --no-bias-swiglu-fusion \
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

msrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    --ai-framework mindspore \
    | tee logs/pretrain_glm4_9b_8k_mcore.log
