#!/bin/bash

# Check the parmeters
if [ $# -ne 1 ]; then
    echo "Usage: bash $0 <step>. Eg: bash $0 step0"
    exit 1
fi

echo "activate conda environment..."
# conda activate depthHelps

param=$1
if [ "$param" == "step0" ]; then
    echo "Execute the step0"
    MODEL_NAME=roboflamingo-mpt_3b_depth
    DATASET_NAME=DepthLiberoDataset
    
    deepspeed main/src.py \
    --model_name=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --run_name=$MODEL_NAME-$DATASET_NAME \
    --checkpoint=RoboFlamingo/models/robo_flamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_4.pth \
    --auto_remove_prev_ckpt \
    --gradient_accumulation_steps=1 \
    --train_micro_batch_size_per_gpu=8 \
    --num_epochs=5

elif [ "$param" == "step1" ]; then
    echo "Execute the step1"

    deepspeed src/pred_depth.py \
    --run_name=runs/pred_depth \
    --checkpoint=runs/roboflamingo-mpt_3b_depth-DepthLiberoDataset/ckpt/global_step22645/mp_rank_00_model_states.pt \
    --gradient_accumulation_steps=1 \
    --train_micro_batch_size_per_gpu=8 \
    --num_epochs=5

elif [ "$param" == "step2" ]; then
    echo "Execute the step2"

    python src/train_vq.py

elif [ "$param" == "step3" ]; then
    echo "Execute the step3"
    MODEL_NAME=roboflamingo-mpt_3b_depth_depth_codebook_ema_finetune
    DATASET_NAME=DepthLiberoDataset

    echo "Convert the checkpoint format..."
    python src/convert_vq_ckpt.py \
    
    deepspeed src/main.py \
    --local_rank=3 \
    --model_name=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --run_name=runs/$MODEL_NAME-$DATASET_NAME \
    --gradient_accumulation_steps=1 \
    --train_micro_batch_size_per_gpu=8 \
    --num_epochs=2 \
    --checkpoint="RoboFlamingo/models/robo_flamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_4.pth#runs/vq/ckpt/merge_ckpt/epoch4.pt" \
    --auto_remove_prev_ckpt \
    --train_type=depth_codebook_ema_finetune

    python src/merge_checkpoint.py

else
  echo "Unrecognized parameter => $param"
fi