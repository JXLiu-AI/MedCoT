#!/bin/bash

# Model: unifiedqa-t5-base
# ==============================================================================
# ==============================Main Result=====================================
# ==============================================================================
# Dataset: R-RAD (closed-end)

# Method: Explanation
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset rad --method Explanation --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-RAD/closed_end/train.json --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset rad --method Explanation --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-RAD/closed_end/test.json --model_path rad_closed_end_experiments/Explanation --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_closed_end_experiments
# Method: Reasoning
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset rad --method Reasoning --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-RAD/closed_end/train.json --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset rad --method Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-RAD/closed_end/test.json --model_path rad_closed_end_experiments/Reasoning --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_closed_end_experiments
# Method: Two-Stage Reasoning
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset rad --method First-Stage_Reasoning --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-RAD/closed_end/train.json --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset rad --method First-Stage_Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-RAD/closed_end/train.json --model_path rad_closed_end_experiments/First-Stage_Reasoning --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset rad --method First-Stage_Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-RAD/closed_end/test.json --model_path rad_closed_end_experiments/First-Stage_Reasoning --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset rad --method Second-Stage_Reasoning --epoch 20 --lr 5e-5 --bs 8 --source_len 512 --target_len 32 --train_text_file_path rad_closed_end_experiments/First-Stage_Reasoning/train.json --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset rad --method Second-Stage_Reasoning --bs 8 --source_len 512 --target_len 32 --text_file_path rad_closed_end_experiments/First-Stage_Reasoning/test.json --model_path rad_closed_end_experiments/Second-Stage_Reasoning --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_closed_end_experiments
# Method: without_R
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset rad --method without_R --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 32 --train_text_file_path /path/to/data/R-RAD/closed_end/train.json --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset rad --method without_R --bs 8 --source_len 512 --target_len 32 --text_file_path /path/to/data/R-RAD/closed_end/test.json --model_path rad_closed_end_experiments/without_R --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_closed_end_experiments

# Dataset: R-SLAKE (closed-end)

# Method: Explanation
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset slake --method Explanation --epoch 300 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-SLAKE/closed_end/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset slake --method Explanation --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-SLAKE/closed_end/test.json --model_path slake_closed_end_experiments/Explanation --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_closed_end_experiments
# Method: Reasoning
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset slake --method Reasoning --epoch 300 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-SLAKE/closed_end/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset slake --method Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-SLAKE/closed_end/test.json --model_path slake_closed_end_experiments/Reasoning --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_closed_end_experiments
# Method: Two-Stage Reasoning
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset slake --method First-Stage_Reasoning --epoch 300 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-SLAKE/closed_end/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset slake --method First-Stage_Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-SLAKE/closed_end/train.json --model_path slake_closed_end_experiments/First-Stage_Reasoning --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset slake --method First-Stage_Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-SLAKE/closed_end/test.json --model_path slake_closed_end_experiments/First-Stage_Reasoning --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset slake --method Second-Stage_Reasoning --epoch 20 --lr 5e-5 --bs 8 --source_len 512 --target_len 32 --train_text_file_path slake_closed_end_experiments/First-Stage_Reasoning/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset slake --method Second-Stage_Reasoning --bs 8 --source_len 512 --target_len 32 --text_file_path slake_closed_end_experiments/First-Stage_Reasoning/test.json --model_path slake_closed_end_experiments/Second-Stage_Reasoning --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_closed_end_experiments
# Method: without_R
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset slake --method without_R --epoch 300 --lr 5e-4 --bs 8 --source_len 512 --target_len 32 --train_text_file_path /path/to/data/R-SLAKE/closed_end/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python closed_end_generate.py --dataset slake --method without_R --bs 8 --source_len 512 --target_len 32 --text_file_path /path/to/data/R-SLAKE/closed_end/test.json --model_path slake_closed_end_experiments/without_R --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_closed_end_experiments

# Dataset: R-RAD (open-end)

# Method: Explanation
CUDA_VISIBLE_DEVICES=0,1,2,3 python open_end_train.py --dataset rad --method Explanation --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-RAD/open_end/train.json --img_file_path /path/to/data/R-RAD/detr.pth -img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset rad --method Explanation --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-RAD/open_end/test.json --model_path rad_open_end_experiments/Explanation --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_open_end_experiments
# Method: Reasoning
CUDA_VISIBLE_DEVICES=0,1,2,3 python open_end_train.py --dataset rad --method Reasoning --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-RAD/open_end/train.json --img_file_path /path/to/data/R-RAD/detr.pth -img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset rad --method Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-RAD/open_end/test.json --model_path rad_open_end_experiments/Reasoning --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_open_end_experiments
# Method: Two-Stage Reasoning
CUDA_VISIBLE_DEVICES=0,1,2,3 python open_end_train.py --dataset rad --method First-Stage_Reasoning --epoch 150 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-RAD/open_end/train.json --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset rad --method First-Stage_Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-RAD/open_end/train.json --model_path rad_open_end_experiments/First-Stage_Reasoning --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset rad --method First-Stage_Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-RAD/open_end/test.json --model_path rad_open_end_experiments/First-Stage_Reasoning --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1,2,3 python open_end_train.py --dataset rad --method Second-Stage_Reasoning --epoch 20 --lr 5e-5 --bs 8 --source_len 512 --target_len 32 --train_text_file_path rad_open_end_experiments/First-Stage_Reasoning/train.json --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir rad_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset rad --method Second-Stage_Reasoning --bs 8 --source_len 512 --target_len 32 --text_file_path rad_open_end_experiments/First-Stage_Reasoning/test.json --model_path rad_open_end_experiments/Second-Stage_Reasoning --img_file_path /path/to/data/R-RAD/detr.pth --img_name_map /path/to/data/R-RAD/name_map.json --output_dir rad_open_end_experiments

# Dataset: R-SLAKE (open-end)

# Method: Explanation
CUDA_VISIBLE_DEVICES=0,1,2,3 python open_end_train.py --dataset slake --method Explanation --epoch 300 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-SLAKE/open_end/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth -img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset slake --method Explanation --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-SLAKE/open_end/test.json --model_path slake_open_end_experiments/Explanation --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_open_end_experiments
# Method: Reasoning
CUDA_VISIBLE_DEVICES=0,1,2,3 python open_end_train.py --dataset slake --method Reasoning --epoch 300 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-SLAKE/open_end/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth -img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset slake --method Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-SLAKE/open_end/test.json --model_path slake_open_end_experiments/Reasoning --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_open_end_experiments
# Method: Two-Stage Reasoning
CUDA_VISIBLE_DEVICES=0,1,2,3 python open_end_train.py --dataset slake --method First-Stage_Reasoning --epoch 300 --lr 5e-4 --bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/data/R-SLAKE/open_end/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset slake --method First-Stage_Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-SLAKE/open_end/train.json --model_path slake_open_end_experiments/First-Stage_Reasoning --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset slake --method First-Stage_Reasoning --bs 8 --source_len 512 --target_len 256 --text_file_path /path/to/data/R-SLAKE/open_end/test.json --model_path slake_open_end_experiments/First-Stage_Reasoning --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1,2,3 python open_end_train.py --dataset slake --method Second-Stage_Reasoning --epoch 20 --lr 5e-5 --bs 8 --source_len 512 --target_len 32 --train_text_file_path slake_open_end_experiments/First-Stage_Reasoning/train.json --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --pretrained_model_path unifiedqa-t5-base --output_dir slake_open_end_experiments
CUDA_VISIBLE_DEVICES=0,1 python open_end_generate.py --dataset slake --method Second-Stage_Reasoning --bs 8 --source_len 512 --target_len 32 --text_file_path slake_open_end_experiments/First-Stage_Reasoning/test.json --model_path slake_open_end_experiments/Second-Stage_Reasoning --img_file_path /path/to/data/R-SLAKE/detr.pth --img_name_map /path/to/data/R-SLAKE/name_map.json --output_dir slake_open_end_experiments
# ==============================================================================
# ==============================================================================
# ==============================================================================


# Model: Gemini-Pro
# ==============================================================================
# =========================Rationale Quality Assessment=========================
# ==============================================================================
for i in {0..4}
do
  output_path="gemini_answer_$i.json"
  python gemini.py --dataset_type rad --file_path rad_closed_end_experiments/without_R/test.json --image_dir /path/to/data/R-RAD/images/ --output_path $output_path
  python gemini.py --solution --dataset_type rad --file_path rad_closed_end_experiments/Explanation/test.json --image_dir /path/to/data/R-RAD/images/ --output_path $output_path
  python gemini.py --solution --dataset_type rad --file_path rad_closed_end_experiments/Reasoning/test.json --image_dir /path/to/data/R-RAD/images/ --output_path $output_path
  python gemini.py --solution --dataset_type rad --file_path rad_closed_end_experiments/Second-Stage_Reasoning/test.json --image_dir /path/to/data/R-RAD/images/ --output_path $output_path
  python gemini.py --dataset_type slake --file_path slake_closed_end_experiments/without_R/test.json --image_dir /path/to/data/R-SLAKE/img/ --output_path $output_path
  python gemini.py --solution --dataset_type slake --file_path slake_closed_end_experiments/Explanation/test.json --image_dir /path/to/data/R-SLAKE/img/ --output_path $output_path
  python gemini.py --solution --dataset_type slake --file_path slake_closed_end_experiments/Reasoning/test.json --image_dir /path/to/data/R-SLAKE/img/ --output_path $output_path
  python gemini.py --solution --dataset_type slake --file_path slake_closed_end_experiments/Second-Stage_Reasoning/test.json --image_dir /path/to/data/R-SLAKE/img/ --output_path $output_path
done
# ==============================================================================
# ==============================================================================
# ==============================================================================
