#!/bin/bash

# RAD_Q_AR
#CUDA_VISIBLE_DEVICES=0,1,2,3 python OpenEndTrain.py --dataset_type rad --answer_first --rational --no_validate --suffix 0 --epoch 150 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-RAD/open-end/trainset.json --test_text_file_path /path/to/VQA-RAD/open-end/testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/OpenExperiments --img_type detr &
# RAD_Q_RA
#CUDA_VISIBLE_DEVICES=4,5,6,7 python OpenEndTrain.py --dataset_type rad --rational --no_validate --suffix 1 --epoch 150 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-RAD/open-end/trainset.json --test_text_file_path /path/to/VQA-RAD/open-end/testset.json --img_file_path /path/to/VQA-RAD/detr.pth --img_name_map /path/to/VQA-RAD/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/OpenExperiments --img_type detr
# RAD_Q_R_A
CUDA_VISIBLE_DEVICES=4,5,6,7 python OpenEndTrain.py --dataset_type rad \
                                                    --caption \
                                                    --no_validate \
                                                      --suffix 0 \
                                                      --epoch 100 \
                                                      --lr 8e-5 \
                                                      --bs 8 \
                                                      --eval_bs 8 \
                                                      --source_len 512 \
                                                      --target_len 256 \
                                                      --train_text_file_path /path/to/VQA-RAD/open-end/train.json \
                                                      --test_text_file_path /path/to/VQA-RAD/open-end/test.json \
                                                      --img_file_path /path/to/VQA-RAD/detr.pth \
                                                      --img_name_map  /path/to/VQA-RAD/name_map.json \
                                                      --pretrained_model_path /path/to/flan-t5-base \
                                                      --img_type detr \
                                                      --output_dir ./experiments/OpenExperiments/R-RAD-Cap-Reverse_open_CoT \
                                                      
# SLAKE_Q_AR
#CUDA_VISIBLE_DEVICES=0,1,2,3 python OpenEndTrain.py --dataset_type slake --answer_first --rational --no_validate --suffix 2 --epoch 300 --lr 5e-4 --bs 8 --eval_bs 8 --source_len 512 --target_len 256 --train_text_file_path /path/to/VQA-SLAKE/open-end/train.json --test_text_file_path /path/to/VQA-SLAKE/open-end/test.json --img_file_path /path/to/VQA-SLAKE/detr.pth --img_name_map /path/to/VQA-SLAKE/name_map.json --pretrained_model_path /path/to/unifiedqa-t5-base --output_dir ./experiments/OpenExperiments --img_type detr &
# SLAKE_Q_RA
# CUDA_VISIBLE_DEVICES=4,5,6,7 python OpenEndTrain.py --dataset_type slake \
#                                                     --rational \
#                                                     --no_validate \
#                                                     --suffix 3 \
#                                                     --epoch 300 \
#                                                     --lr 5e-4 \
#                                                     --bs 8 \
#                                                     --eval_bs 8 \
#                                                     --source_len 512 \
#                                                     --target_len 256 \
#                                                     --train_text_file_path /path/to/VQA-SLAKE/open-end/train.json \
#                                                     --test_text_file_path /path/to/VQA-SLAKE/open-end/test.json \
#                                                     --img_file_path /path/to/VQA-SLAKE/detr.pth \
#                                                     --img_name_map /path/to/VQA-SLAKE/name_map.json \
#                                                     --pretrained_model_path /path/to/unifiedqa-t5-base \
#                                                     --output_dir ./experiments/OpenExperiments \
#                                                     --img_type detr
# CUDA_VISIBLE_DEVICES=4,5,6,7 python OpenEndTrain_R.py --dataset_type slake \
#                                                       --suffix 5 \
#                                                       --epoch 300 \
#                                                       --lr 5e-4 \
#                                                       --bs 8 \
#                                                       --eval_bs 8 \
#                                                       --source_len 512 \
#                                                       --target_len 256 \
#                                                       --train_text_file_path /path/to/VQA-SLAKE/open-end/train.json \
#                                                       --test_text_file_path /path/to/VQA-SLAKE/open-end/test.json \
#                                                       --img_file_path /path/to/VQA-SLAKE/detr.pth \
#                                                       --img_name_map /path/to/VQA-SLAKE/name_map.json \
#                                                       --pretrained_model_path /path/to/unifiedqa-t5-base \
#                                                       --output_dir ./experiments/OpenExperiments \
#                                                       --img_type detr
