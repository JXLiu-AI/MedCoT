#!/bin/bash

# Model Cards:
# /path/to/unifiedqa-t5-base
# /path/to/t5_large
# /path/to/flan-t5-large

# Model: Unifiedqa-t5-base
# Dataset: RAD
# Action: Train Model (->Answer) Caption
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type rad --caption --suffix 0 --epoch 150 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 128 --train_text_file_path /path/to/VQA-RAD/cap_trainset.json --test_text_file_path /path/to/VQA-RAD/cap_testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr
# Action: Train Model (->Rationale+Answer) Caption
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type rad --rational --caption --suffix 1 --epoch 150 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 128 --train_text_file_path /path/to/VQA-RAD/cap_trainset.json --test_text_file_path /path/to/VQA-RAD/cap_testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr
# Action: Train Model (->Answer) No Caption
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type rad --suffix 2 --epoch 150 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 128 --train_text_file_path /path/to/VQA-RAD/cap_trainset.json --test_text_file_path /path/to/VQA-RAD/cap_testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr
# Action: Train Model (->Rationale+Answer) No Caption
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type rad --rational --suffix 3 --epoch 150 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 128 --train_text_file_path /path/to/VQA-RAD/cap_trainset.json --test_text_file_path /path/to/VQA-RAD/cap_testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr

# Model: Unifiedqa-t5-base
# Dataset: SLAKE
# Action: Train Model (->Answer) Caption
#CUDA_VISIBLE_DEVICES=0,1,2,3 python DirectAnswer.py --dataset_type slake --caption --suffix 4 --epoch 150 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 64 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr &
# Action: Train Model (->Answer) No Caption
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type slake --suffix 5 --epoch 150 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 64 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr
#wait
# Action: Train Model (->Rationale+Answer) Caption
#CUDA_VISIBLE_DEVICES=0,1,2,3 python DirectAnswer.py --dataset_type slake --rational --caption --suffix 6 --epoch 150 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr &
# Action: Train Model (->Rationale+Answer) No Caption
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type slake --rational --suffix 7 --epoch 150 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr

# Model: Unifiedqa-t5-base
# Dataset: RAD and SLAKE
# ACtion: Train Model (->Rationale+Answer) No Caption
#CUDA_VISIBLE_DEVICES=0,1,2,3 python DirectAnswer.py --dataset_type slake --rational --suffix 8 --epoch 300 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr &
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type rad --rational --suffix 9 --epoch 300 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-RAD/cap_trainset.json --test_text_file_path /path/to/VQA-RAD/cap_testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr

# Model: Unifiedqa-t5-base
# Dataset: SLAKE
# Action: Train Model (->Rationale+Answer) larger lr
#CUDA_VISIBLE_DEVICES=0,1,2,3 python DirectAnswer.py --dataset_type slake --rational --suffix 10 --epoch 300 --lr 7e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr &
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type slake --rational --suffix 11 --epoch 300 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr

# Model: Unifiedqa-t5-base
# Motivate: test epoch and lr for new solution
# Dataset: RAD (V2)
# Action: Train Model (->Rationale+Answer) No Caption
#CUDA_VISIBLE_DEVICES=0,1,2,3 python DirectAnswer.py --dataset_type rad --rational --suffix 12 --epoch 300 --lr 5e-5 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-RAD/V2/trainset.json --test_text_file_path /path/to/VQA-RAD/V2/testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr &
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type rad --rational --suffix 13 --epoch 300 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-RAD/V2/trainset.json --test_text_file_path /path/to/VQA-RAD/V2/testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr

# Model: Unifiedqa-t5-base
# Dataset: SLAKE
# Action: Train Model (->Rationale+Answer) Caption
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type slake --caption --rational --suffix 14 --epoch 300 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr

## Model: Unifiedqa-t5-base
## Dataset: SLAKE
## Action: Train Model (->Answer+Rationale) No Caption
#CUDA_VISIBLE_DEVICES=0,1,2,3 python DirectAnswer.py --dataset_type slake --answer_first --rational --suffix 15 --epoch 300 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr &
## Model: Unifieqa-t5-base
## Dataset: RAD
## Action: Train Model (->Answer+Rationale) No Caption
#CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type rad --answer_first --rational --suffix 16 --epoch 300 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-RAD/V2/cap_trainset.json --test_text_file_path /path/to/VQA-RAD/V2/cap_testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr

# ModelL Unifiedqa-t5-base
# Dataset: RAD (V2) and SLAKE
# Action: Train Model (->Answer) No Caption
CUDA_VISIBLE_DEVICES=0,1,2,3 python DirectAnswer.py --dataset_type rad --suffix 17 --epoch 300 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 32 --train_text_file_path /path/to/VQA-RAD/V2/cap_trainset.json --test_text_file_path /path/to/VQA-RAD/V2/cap_testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr &
CUDA_VISIBLE_DEVICES=4,5,6,7 python DirectAnswer.py --dataset_type slake --suffix 18 --epoch 300 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 32 --train_text_file_path /path/to/VQA-SLAKE/cap_train.json --test_text_file_path /path/to/VQA-SLAKE/cap_test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/DirectAnswer/ --img_type detr
