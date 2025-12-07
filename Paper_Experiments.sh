# CUDA_VISIBLE_DEVICES=0,3,4,7 python MyTrain.py --dataset_type rad \
#                                                --suffix 0 \
#                                                --epoch 100 \
#                                                --lr 8e-5 \
#                                                --bs 8 \
#                                                --eval_bs 8 \
#                                                --source_len 512 \
#                                                --target_len 32 \
#                                                --train_text_file_path /path/to/VQA-RAD/Step1_RAD_closed_train.json \
#                                                --test_text_file_path /path/to/VQA-RAD/Step1_RAD_closed_test.json \
#                                                --img_file_path /path/to/VQA-RAD/detr.pth \
#                                                --img_name_map /path/to/VQA-RAD/name_map.json \
#                                                --pretrained_model_path /path/to/flan-t5-base \
#                                                --img_type detr \
#                                                --output_dir ./experiments/TwoStage/R-RAD-Cap-Reverse111111_ab_nothing

# CUDA_VISIBLE_DEVICES=0,3,4,7 python MyTrain.py --dataset_type slake \
#                                                --suffix 0 \
#                                                --epoch 100 \
#                                                --lr 8e-5 \
#                                                --bs 8 \
#                                                --eval_bs 8 \
#                                                --source_len 512 \
#                                                --target_len 32 \
#                                                --train_text_file_path /path/to/VQA-SLAKE/Step1_SLAKE_closed_train.json \
#                                                --test_text_file_path /path/to/VQA-SLAKE/Step1_SLAKE_closed_test.json \
#                                                --img_file_path /path/to/VQA-SLAKE/detr.pth \
#                                                --img_name_map /path/to/VQA-SLAKE/name_map.json \
#                                                --pretrained_model_path /path/to/flan-t5-base \
#                                                --img_type detr \
#                                                --output_dir ./experiments/TwoStage/R-SLAKE-Cap-Reverse111111_ab_nothing

CUDA_VISIBLE_DEVICES=0,3,4,7 python MyTrain.py --dataset_type rad2019 \
                                                --caption \
                                               --suffix 0 \
                                               --epoch 100 \
                                               --lr 8e-5 \
                                               --bs 8 \
                                               --eval_bs 8 \
                                               --source_len 512 \
                                               --target_len 32 \
                                               --train_text_file_path /path/to/VQA-2019/Step1_ImageClef_closed_train.json \
                                               --test_text_file_path /path/to/VQA-2019/Step1_ImageClef_closed_test.json \
                                               --img_file_path /path/to/VQA-2019/detr.pth \
                                               --img_name_map /path/to/VQA-2019/name_map.json \
                                               --pretrained_model_path /path/to/flan-t5-base \
                                               --img_type detr \
                                               --output_dir ./experiments/TwoStage/R-2019-Cap-Reverse111111_ab_nothing

# CUDA_VISIBLE_DEVICES=7,4,5,6 python MyTrain.py --dataset_type rad \
#                                                --caption \
#                                                --suffix 0 \
#                                                --epoch 100 \
#                                                --lr 8e-5 \
#                                                --bs 8 \
#                                                --eval_bs 8 \
#                                                --source_len 512 \
#                                                --target_len 32 \
#                                                --train_text_file_path /path/to/VQA-RAD/Step2_RAD_closed_train.json \
#                                                --test_text_file_path /path/to/VQA-RAD/Step2_RAD_closed_test.json \
#                                                --img_file_path /path/to/VQA-RAD/detr.pth \
#                                                --img_name_map /path/to/VQA-RAD/name_map.json \
#                                                --pretrained_model_path /path/to/flan-t5-base \
#                                                --img_type detr \
#                                                --output_dir ./experiments/TwoStage/R-RAD-Cap-Reverse222222_ab_6_1



# CUDA_VISIBLE_DEVICES=7,4,5,6 python MyTrain.py --dataset_type slake \
#                                                --caption \
#                                                --suffix 0 \
#                                                --epoch 100 \
#                                                --lr 8e-5 \
#                                                --bs 8 \
#                                                --eval_bs 8 \
#                                                --source_len 512 \
#                                                --target_len 32 \
#                                                --train_text_file_path /path/to/VQA-SLAKE/Step1_SLAKE_closed_train.json \
#                                                --test_text_file_path /path/to/VQA-SLAKE/Step1_SLAKE_closed_test.json \
#                                                --img_file_path /path/to/VQA-SLAKE/detr.pth \
#                                                --img_name_map /path/to/VQA-SLAKE/name_map.json \
#                                                --pretrained_model_path /path/to/flan-t5-base \
#                                                --img_type detr \
#                                                --output_dir ./experiments/TwoStage/R-SLAKE-Cap-Reverse111111_ab

# CUDA_VISIBLE_DEVICES=7,4,5,6 python MyTrain.py --dataset_type slake \
#                                                --caption \
#                                                --suffix 0 \
#                                                --epoch 100 \
#                                                --lr 8e-5 \
#                                                --bs 8 \
#                                                --eval_bs 8 \
#                                                --source_len 512 \
#                                                --target_len 32 \
#                                                --train_text_file_path /path/to/VQA-SLAKE/Step2_SLAKE_closed_train.json \
#                                                --test_text_file_path /path/to/VQA-SLAKE/Step2_SLAKE_closed_test.json \
#                                                --img_file_path /path/to/VQA-SLAKE/detr.pth \
#                                                --img_name_map /path/to/VQA-SLAKE/name_map.json \
#                                                --pretrained_model_path /path/to/flan-t5-base \
#                                                --img_type detr \
