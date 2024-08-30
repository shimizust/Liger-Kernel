USE_LIGER_VALUES=("False")
BATCH_SIZE_VALUES=(192)
# MODEL_NAME_PREFIX="gemma"
# MODEL_PATH="/shared/public/models/gemma-7b-it"
# MODEL_NAME_PREFIX="phi3"
# MODEL_PATH="/shared/public/models/microsoft/Phi-3.5-mini-instruct"
MODEL_NAME_PREFIX="mistral"
MODEL_PATH="/shared/public/models/Mistral-7B"
# MODEL_NAME_PREFIX="llama"
# MODEL_PATH="/shared/public/models/Meta-Llama-3-8B"


for USE_LIGER in "${USE_LIGER_VALUES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZE_VALUES[@]}"; do
        echo "Running with use_liger=$USE_LIGER and batch_size=$BATCH_SIZE"

        LOG_FILE="${MODEL_NAME_PREFIX}_use_liger_${USE_LIGER}_batch_size_${BATCH_SIZE}.log"

        torchrun --nnodes=1 --nproc-per-node=4 training.py \
            --bf16 \
            --num_train_epochs 1 \
            --max_steps 20 \
            --model_name $MODEL_PATH \
            --dataset "/shared/public/data/tatsu-lab" \
            --per_device_train_batch_size $BATCH_SIZE \
            --per_device_eval_batch_size 64 \
            --eval_strategy "no" \
            --save_strategy "no" \
            --learning_rate 6e-6 \
            --weight_decay 0.05 \
            --warmup_ratio 0.1 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --include_num_input_tokens_seen \
            --report_to none \
            --fsdp "full_shard auto_wrap" \
            --fsdp_config config/fsdp_config.json \
            --seed 42 \
            --use_liger $USE_LIGER \
            --output_dir alpaca_finetuning \
            > $LOG_FILE

        sleep 5
    done
done