import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class DirectAnswerDataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _rational, _caption, source_len, target_len):
        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        self.source_len = source_len
        self.target_len = target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)
        for problem in data:
            pair = InputAndTargetAndImg(data[problem])
            prompt = pair.get_input(_rational, _caption)
            target = pair.get_target(_rational)
            img = pair.get_img(_dataset="rad2019")
            self.source_text.append(prompt)
            self.target_text.append(target)
            self.img_index.append(int(name_map[img]))
            self.problem_id.append(problem)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])
        img_index = self.img_index[item]

        # Normalize whitespace: remove extra spaces, tabs, and newlines.
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        image_ids = self.pretrained_feature[img_index].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids
        }

class VqaRadDataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _rational, _caption, source_len, target_len):

        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        self.source_len = source_len
        self.target_len = target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)

        for problem in data:
            pair = InputAndTargetAndImg(data[problem])
            prompt = pair.get_input(_rational, _caption)
            target = pair.get_target(_rational)
            img = pair.get_img(_dataset="rad")
            self.source_text.append(prompt)
            self.target_text.append(target)
            self.img_index.append(int(name_map[img]))
            self.problem_id.append(problem)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])
        img_index = self.img_index[item]

        # Normalize whitespace: remove extra spaces, tabs, and newlines.
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        image_ids = self.pretrained_feature[img_index].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids
        }

class SegVqaRadDataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _rational, _caption, source_len, target_len):

        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        self.source_len = source_len
        self.target_len = target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)

        for problem in data:
            pair = InputAndTargetAndImg(data[problem])
            prompt = pair.get_input(_rational, _caption)
            target = pair.get_target(_rational)
            img = pair.get_img(_dataset="rad")
            self.source_text.append(prompt)
            self.target_text.append(target)
            self.img_index.append(int(name_map[f"{problem}_{img}"]))
            self.problem_id.append(problem)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])
        img_index = self.img_index[item]

        # Normalize whitespace: remove extra spaces, tabs, and newlines.
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        image_ids = self.pretrained_feature[img_index].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids
        }

class VqaSlakeDataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _rational, _caption, source_len, target_len):
        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        self.source_len = source_len
        self.target_len = target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)

        for problem in data:
            pair = InputAndTargetAndImg(data[problem])
            prompt = pair.get_input(_rational, _caption)
            target = pair.get_target(_rational)
            img = pair.get_img(_dataset="slake")
            self.source_text.append(prompt)
            self.target_text.append(target)
            self.img_index.append(int(name_map[str(img)]))
            self.problem_id.append(problem)

    def __len__(self):
        return len(self.source_text)

    def __getitem__(self, item):
        source_text = str(self.source_text[item])
        target_text = str(self.target_text[item])
        img_index = self.img_index[item]

        # Normalize whitespace: remove extra spaces, tabs, and newlines.
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        image_ids = self.pretrained_feature[img_index].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids
        }


class InputAndTargetAndImg:
    def __init__(self, problem):
        self.problem = problem
        self.options = ['A', 'B']
        self.question_text = self.get_question_text()
        self.answer_text = self.get_answer()
        self.choice_text = self.get_choice_text()
        self.solution_text = self.get_solution_text()

    def get_choice_text(self):
        choices = self.problem['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(self.options[i], c))
        choice_txt = " ".join(choice_list)
        return choice_txt

    def get_question_text(self):
        return self.problem['question']
        # return self.problem['sent']  #for pathvqa

    def get_answer(self):
        return "("+self.options[self.problem['answer']]+")"

    def get_solution_text(self):
        return self.problem['solution']
    
    def get_target(self, _rational=True):
        if _rational:
            return f"{self.problem['solution']}"
        else:
            return f"The answer is {self.answer_text}."

    def get_input(self, _rational=True, _caption=False):
        prior_text = f"Question: {self.question_text}\nOptions: {self.choice_text}\n"
        if _caption:
            prior_text = prior_text+f"Caption: {self.problem['caption']}\n"
        if _rational:
            return prior_text+f"Solution:"
        else:
            return prior_text+f"Solution: {self.solution_text}\nAnswer:"

    def get_img(self, _dataset):
        if _dataset == "rad":
            # import pdb;pdb.set_trace()
            return self.problem['image'][:-4]
        elif _dataset == "slake":
            return self.problem["img_id"]
        elif _dataset == "rad2019":
            # import pdb;pdb.set_trace()
            return self.problem["image"][:-4]  
            # return self.problem["img_id"][:-4]   #for pathvqa
        else:
            raise ValueError(f"Invalid _dataset value: {_dataset}. The value must be 'rad' or 'slake'.")


if __name__ == "__main__":
    # text_file_path = "/home/chenyizhou/mm_cot/VQA-SLAKE_ByUs/without_open/train.json"
    # img_file_path = "/home/chenyizhou/mm_cot/VQA-SLAKE_ByUs/detr.pth"
    # img_name_map = "/home/chenyizhou/mm_cot/VQA-SLAKE_ByUs/name_map.json"
    # tokenizer = AutoTokenizer.from_pretrained("/home/chenyizhou/mm_cot/src/unifiedqa-t5-base")
    # train_set = VqaSlakeDataset(
    #     _tokenizer=tokenizer,
    #     _text_file_path=text_file_path,
    #     _img_file_path=img_file_path,
    #     _img_name_map=img_name_map,
    #     _rational=True,
    #     source_len=512,
    #     target_len=512
    # )
    # train = DataLoader(train_set, batch_size=1, shuffle=False)
    # for idx, i in enumerate(train):
    #     a = tokenizer.batch_decode(i['labels'], skip_special_tokens=True)
    #     print(a)
    #     break
    
    text_file_path = "/home/chenyizhou/mm_cot/VQA-RAD_ByUs/without_open/cap_trainset.json"
    img_file_path = "/home/chenyizhou/mm_cot/VQA-RAD_ByUs/without_open/detr.pth"
    img_name_map = "/home/chenyizhou/mm_cot/VQA-RAD_ByUs/without_open/name_map.json"
    tokenizer = AutoTokenizer.from_pretrained("/home/chenyizhou/mm_cot/src/unifiedqa-t5-base")
    train_set = VqaRadDataset(
        _tokenizer=tokenizer,
        _text_file_path=text_file_path,
        _img_file_path=img_file_path,
        _img_name_map=img_name_map,
        _rational=True,
        _caption=False,
        source_len=512,
        target_len=512
    )
    for i in train_set.source_text:
        print(i)
