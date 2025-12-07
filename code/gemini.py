import google.generativeai as genai
import json
from PIL import Image
import os
import re
import time
import random
import argparse
import base64
import requests


gemini_api_key = [
    'YOUR_API_KEY_HERE',  # KEY_1
    'YOUR_API_KEY_HERE',  # KEY_2
]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_ans(_ans):
    pattern = re.compile(r'\(([A-B])\)')
    res = pattern.findall(_ans)

    if len(res) == 1:
        _answer = res[0]  # 'A', 'B', ...
    else:
        _answer = "FAILED"
    return _answer

def load_existing_results(_file_path):
    if os.path.exists(_file_path):
        with open(_file_path, 'r', encoding='utf-8') as File:
            return json.load(File)
    else:
        return None

def genimi_to_answer(_solution, _file_path, _image_dir, _output_path, _dataset_type):
    def get_headers():
        return random.randint(0, len(gemini_api_key)-1)

    existing_results = load_existing_results(_output_path)
    if existing_results:
        result = existing_results
    else:
        result = {}
    with open(_file_path, 'r', encoding='utf-8') as File:
        data = json.load(File)
    print(f"There are {len(data)-len(result)} questions.\n")
    instruction = "Let's think step-by-step, and please answer the following VQA question:\n"
    for num, item in enumerate(data):
        if existing_results and f"question_{item}" in existing_results:
            continue
        question = data[item]['question']
        choices = f"(A) yes (B) no"

        context = f"Question: {question}\n"
        if _solution:
            context = context + f"Solution: {data[item]['solution']}\n"
        context = context + f"Options: {choices}\nAnswer:"

        if _dataset_type == 'rad':
            image_path = os.path.join(_image_dir, data[item]['image'])
        elif _dataset_type == 'slake':
            image_path = os.path.join(_image_dir, data[item]['img_name'])
        else:
            raise ValueError
        print(f"Image: {image_path}\n{context}")
        image = Image.open(image_path).convert('RGB')
        while True:
            try:
                genai.configure(api_key=gemini_api_key[get_headers()], transport='rest')
                model = genai.GenerativeModel('gemini-pro-vision')
                response = model.generate_content(
                    [image, instruction+context], stream=False,
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                         },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
                )
                response.resolve()
                print(f"Output: {response.parts[0].text}")
                answer = extract_ans(response.parts[0].text)
                result[f"question_{item}"] = f"The answer is ({answer})."
                with open(_output_path, 'w', encoding='utf-8') as OutFile:
                    json.dump(result, OutFile, indent=4, ensure_ascii=False)
                print(f"{num}: question_{item}: {answer}.\n")
                break

            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                time.sleep(30)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution', action='store_true', default=False)
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--dataset_type', type=str)
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    genimi_to_answer(
        _solution=args.solution,
        _file_path=args.file_path,
        _image_dir=args.image_dir,
        _output_path=os.path.join(os.path.dirname(args.file_path), args.output_path),
        _dataset_type=args.dataset_type
    )
