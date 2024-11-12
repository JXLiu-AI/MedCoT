#!/bin/bash

python extract_img_feature.py --device cuda:0 --image_dir data/RAD/images/ --output_dir  data/RAD/
python extract_img_feature.py --device cuda:0 --image_dir data/SLAKE/img/ --output_dir  data/SLAKE/

