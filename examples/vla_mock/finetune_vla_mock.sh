#!/bin/bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

MM_DATA="./examples/vla_mock/data_vla_mock.json"
MM_MODEL="./examples/vla_mock/model_vla_mock.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH=""
SAVE_PATH="save_dir_vla_mock"
OXE_DATA_ROOT="../unifolm-vla-main/data/modified_libero_rlds"
DATA_MIX="libero_4_task_no_noops"

export OXE_DATA_ROOT
export DATA_MIX
export MM_DATA
export VLA_ROBOT_PLATFORM=LIBERO
export VLA_DATA_ROOT_DIR="${OXE_DATA_ROOT}"
export VLA_DATA_MIX="${DATA_MIX}"

TP=${TP:-1}
PP=${PP:-2}
CP=${CP:-1}
NUM_LAYERS=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(c['text_decoder']['num_layers'])")
HIDDEN_SIZE=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(c['text_decoder']['hidden_size'])")
FFN_HIDDEN_SIZE=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(c['text_decoder']['ffn_hidden_size'])")
NUM_ATTENTION_HEADS=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(c['text_decoder']['num_attention_heads'])")
GROUP_QUERY_ATTENTION=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(int(bool(c['text_decoder'].get('group_query_attention', False))))")
NUM_QUERY_GROUPS=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(int(c['text_decoder'].get('num_query_groups', 1)))")
VOCAB_SIZE=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(c['text_decoder']['vocab_size'])")
SEQ_LENGTH=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(c['text_decoder']['seq_length'])")
MAX_POSITION_EMBEDDINGS=$(python -c "import json; c=json.load(open('${MM_MODEL}','r',encoding='utf-8')); print(c['text_decoder']['max_position_embeddings'])")
if [ $((NUM_ATTENTION_HEADS % TP)) -ne 0 ]; then
  echo "Invalid TP=${TP}: num_attention_heads=${NUM_ATTENTION_HEADS} must be divisible by TP."
  exit 1
fi
EFFECTIVE_NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS}
if [ "${GROUP_QUERY_ATTENTION}" -eq 1 ] && [ $((NUM_QUERY_GROUPS % TP)) -ne 0 ]; then
  EFFECTIVE_NUM_QUERY_GROUPS=${NUM_ATTENTION_HEADS}
fi
if [ "${GROUP_QUERY_ATTENTION}" -eq 1 ] && [ $((EFFECTIVE_NUM_QUERY_GROUPS % TP)) -ne 0 ]; then
  echo "Invalid TP=${TP}: num_query_groups=${NUM_QUERY_GROUPS} (or fallback=${EFFECTIVE_NUM_QUERY_GROUPS}) must be divisible by TP."
  exit 1
fi
MBS=1
GRAD_ACC_STEP=4
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ulysses_cp_algo \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --tokenizer-type NullTokenizer \
    --vocab-size ${VOCAB_SIZE} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 0 \
    --train-iters 20 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --seed 42 \
    --bf16 \
    --use-flash-attn \
    --use-distributed-optimizer \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --num-workers 4 \
    --distributed-timeout-minutes 20 \
"

if [ -n "$LOAD_PATH" ]; then
  GPT_ARGS="${GPT_ARGS} --load ${LOAD_PATH}"
fi
if [ "${GROUP_QUERY_ATTENTION}" -eq 1 ]; then
  GPT_ARGS="${GPT_ARGS} --group-query-attention --num-query-groups ${EFFECTIVE_NUM_QUERY_GROUPS}"
fi

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 20 \
    --eval-interval 10 \
    --eval-iters 2 \
    --save $SAVE_PATH \
    --ckpt-format torch \
    --log-tps \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS pretrain_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_vla_mock_${logfile}.log
