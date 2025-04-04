#!/bin/bash

# Set NCCL environment variables for better distributed training
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_P2P_DISABLE=1

# Create results directory with timestamp
results_dir="experiment_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$results_dir"

# Start timestamp and log it
start_time=$(date +%s)
echo "Starting experiments at $(date)" | tee "${results_dir}/experiment.log"

# Model configurations
declare -a model_sizes=("small" "large")
declare -a tokenizations=("char" "gpt2")
declare -a sampling_strategies=("top_k")
declare -a temperatures=(0.7)

# Training phase
echo "Starting training phase..." | tee -a "${results_dir}/experiment.log"
for model_size in "${model_sizes[@]}"; do
    for tokenization in "${tokenizations[@]}"; do
        echo "Training ${model_size} model with ${tokenization} tokenization..." | tee -a "${results_dir}/experiment.log"
        
        # Create checkpoint directory
        checkpoint_dir="${results_dir}/checkpoints/${model_size}_${tokenization}"
        mkdir -p "$checkpoint_dir"
        
        # Log training start
        training_log="${results_dir}/training_${model_size}_${tokenization}.log"
        echo "Starting training at $(date)" > "$training_log"
        
        # Train model with output directory
        CUDA_VISIBLE_DEVICES=0,1,2 python3 shakespeare_chat_bot.py \
            --model_size "$model_size" \
            --tokenization "$tokenization" \
            --mode train \
            --base_batch_size 32 \
            --max_epochs 35 \
            --output_dir "$results_dir" 2>&1 | tee -a "$training_log"
        
        if [ $? -ne 0 ]; then
            echo "Error: Training failed for ${model_size} model with ${tokenization} tokenization" | tee -a "${results_dir}/experiment.log"
            continue
        fi
        
        echo "Completed training at $(date)" | tee -a "$training_log" "${results_dir}/experiment.log"
    done
done


# Calculate total execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

# Generate experiment summary
{
    echo -e "\nExperiment Summary"
    echo "==================="
    echo "Start time: $(date -d @${start_time})"
    echo "End time: $(date -d @${end_time})"
    echo "Total duration: ${hours}h ${minutes}m ${seconds}s"
    echo "Models tested: ${model_sizes[*]}"
    echo "Tokenizations: ${tokenizations[*]}"
    echo "Sampling strategies: ${sampling_strategies[*]}"
    echo "Temperatures: ${temperatures[*]}"
} | tee "${results_dir}/summary.txt" | tee -a "${results_dir}/experiment.log"

echo "Experiments completed at $(date)" | tee -a "${results_dir}/experiment.log"
echo "Results saved in ${results_dir}" | tee -a "${results_dir}/experiment.log"
echo "Total execution time: ${hours}h ${minutes}m ${seconds}s" | tee -a "${results_dir}/experiment.log"
