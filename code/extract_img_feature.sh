#!/bin/bash

python extract_img_feature.py --device cuda:0 --image_dir /path/to/data/R-RAD/images/ --output_dir  /path/to/data/R-RAD/
python extract_img_feature.py --device cuda:0 --image_dir /path/to/data/R-SLAKE/img/ --output_dir  /path/to/data/R-SLAKE/

