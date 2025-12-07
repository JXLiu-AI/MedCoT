import torch
from PIL import Image
import torchvision.transforms as T
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import argparse
import json

def get_model(_img_type):
    print(f"Loading Model")
    cooelf_path = "/home/.cache/torch/hub/cooelf_detr_main"
    _model = torch.hub.load(cooelf_path, 'detr_resnet101_dc5', pretrained=True, source='local')
    _transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return _model, _transform

def extract_features(_model, _transform, _input_image, _img_type, _device):
        print(f"Loading {_input_image}...")
        img = Image.open(_input_image).convert("RGB")
        input = _transform(img).unsqueeze(0).to(_device)
        with torch.no_grad():
            return _model(input)[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dataset", type=str, choices=["rad", "slake"])
    args = vars(parser.parse_args())
    for arg, value in args.items():
        print(f"{arg}: {value}")

    if args["dataset"] == "rad":
        device = args['device'] if torch.cuda.is_available() else "cpu"
        img_type = args['img_type']
        images_dir = args['image_dir']
        images_path = os.listdir(images_dir)
        output_file_path = args['output_dir']
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        print(f"There are {len(images_path)} images in {images_dir}.")

        model, transform = get_model(img_type)
        model.to(device)
        model.eval()

        name_map = {}   # "KEY" is the image's ID, "VALUE" is the image's Index in the matrix
        vision_features = []
        cnt = 0
        for idx, image_path in enumerate(images_path):
            if image_path.endswith('.jpg'):
                image = os.path.join(images_dir, image_path)
                feature = extract_features(model, transform, image, img_type, device)
                name_map[image_path[:-4]] = str(cnt)
                vision_features.append(feature.detach().cpu())
                cnt += 1

        vision_features = torch.cat(vision_features)
        print(vision_features.shape)
        torch.save(vision_features, os.path.join(output_file_path, f"{img_type}.pth"))
        with open(os.path.join(output_file_path, "name_map.json"), 'w') as outfile:
            json.dump(name_map, outfile, indent=4, ensure_ascii=False)
        print("Files have been written...")
    elif args["dataset"] == "slake":
        device = args['device'] if torch.cuda.is_available() else "cpu"
        img_type = args['img_type']
        images_dir = args['image_dir']
        images_path = os.listdir(images_dir)
        output_file_path = args['output_dir']
        print(images_path)
        print(f"There are {len(images_path)} images in {images_dir}.")

        model, transform = get_model(img_type)
        model.to(device)
        model.eval()

        name_map = {}   # "KEY" is the image's ID, "VALUE" is the image's Index in the matrix
        vision_features = []
        cnt = 0
        for idx, image_path in enumerate(images_path):
            if image_path.endswith('.jpg'):
                image = os.path.join(images_dir, image_path, 'source.jpg')
                feature = extract_features(model, transform, image, img_type, device)
                name_map[image_path[5:]] = str(cnt)
                vision_features.append(feature.detach().cpu())
                cnt += 1

        vision_features = torch.cat(vision_features)
        print(vision_features.shape)
        torch.save(vision_features, output_file_path + f"{img_type}.pth")
        with open(output_file_path+"name_map.json", 'w') as outfile:
            json.dump(name_map, outfile, indent=4, ensure_ascii=False)
        print("Files have been written...")
    else:
        raise ValueError(f"Invalid dataset value: {args['dataset']}. The value must be 'rad' or 'slake'.")
