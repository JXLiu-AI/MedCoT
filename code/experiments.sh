
# # Dataset: RAD (closed-end)
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_train.py --dataset rad --epoch 100 --lr 8e-5 --bs 8 --source_len 512 --target_len 32 --train_text_file_path data/RAD/closed_end/train.json --img_file_path data/RAD/detr.pth --img_name_map data/R-RAD/name_map.json --pretrained_model_path Flan-Base --output_dir rad_closed_end_experiments
CUDA_VISIBLE_DEVICES=0,1,2,3 python closed_end_generate.py --dataset rad --bs 8 --source_len 512 --target_len 32 --text_file_path data/RAD/closed_end/test.json --model_path rad_closed_end_experiments/ --img_file_path data/RAD/detr.pth --img_name_map data/RAD/name_map.json --output_dir rad_closed_end_experiments

