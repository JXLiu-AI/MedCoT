import json
import torch
from torch.utils.data import Dataset

class ClosedMedVQADataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _method, _source_len, _target_len, _dataset):
        self.tokenizer = _tokenizer
        self.source_text = []
        self.target_text = []
        self.problem_id = []
        self.img_index = []
        self.pretrained_feature = torch.load(_img_file_path)
        self.source_len = _source_len
        self.target_len = _target_len

        with open(_text_file_path, "r", encoding="utf-8") as TextFile:
            data = json.load(TextFile)
        with open(_img_name_map, "r", encoding="utf-8") as NameFile:
            name_map = json.load(NameFile)

        for problem in data:
            pair = ClosedInputAndTargetAndImg(data[problem])
            prompt = pair.get_input(method=_method)
            target = pair.get_target(method=_method)
            img = pair.get_img(dataset=_dataset)
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

class ClosedInputAndTargetAndImg:
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

    def get_answer(self):
        return "(" + self.options[self.problem['answer']] + ")"

    def get_solution_text(self):
        return self.problem['solution']

    def get_target(self, method):
        if method == "Explanation":
            return f"The answer is {self.answer_text}.\nSolution: {self.problem['solution']}"
        elif method == "Reasoning":
            return f"{self.problem['solution']}\nAnswer: The answer is {self.answer_text}."
        elif method == "First-Stage_Reasoning":
            return f"{self.problem['solution']}"
        elif method == "Second-Stage_Reasoning":
            return f"The answer is {self.answer_text}."
        elif method == "without_R":
            return f"The answer is {self.answer_text}."
        else:
            raise ValueError("Please Check the Value of Method.")

    def get_input(self, method):
        if method == "Explanation":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nAnswer:"
        elif method == "Reasoning":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nSolution:"
        elif method == "First-Stage_Reasoning":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nSolution:"
        elif method == "Second-Stage_Reasoning":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nSolution: {self.solution_text}\nAnswer:"
        elif method == "without_R":
            return f"Question: {self.question_text}\nOptions: {self.choice_text}\nAnswer:"
        else:
            raise ValueError("Please Check the Value of Method.")

    def get_img(self, dataset):
        if dataset == "rad":
            return self.problem['image'][:-4]
        elif dataset == "slake":
            return self.problem["img_id"]
        else:
            raise ValueError(f"Invalid _dataset value: {dataset}. The value must be 'rad' or 'slake'.")

class OpenMedVQADataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _method, _dataset, _source_len, _target_len):
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
            pair = OpenInputAndTargetAndImg(data[problem])
            prompt = pair.get_input(method=_method)
            target = pair.get_target(method=_method)
            img = pair.get_img(_dataset=_dataset)
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

class OpenInputAndTargetAndImg:
    def __init__(self, problem):
        self.problem = problem
        self.question_text = self.get_question_text()
        self.answer_text = self.get_answer()
        self.solution_text = self.get_solution_text()

    def get_question_text(self):
        return self.problem['question']

    def get_answer(self):
        return self.problem['choices'][0]

    def get_solution_text(self):
        return self.problem['solution']

    def get_target(self, method):
        if method == "Explanation":
            return f"The answer is {self.answer_text}.\nSolution: {self.problem['solution']}"
        elif method == "Reasoning":
            return f"{self.problem['solution']}\nAnswer: The answer is {self.answer_text}."
        elif method == "First-Stage_Reasoning":
            return f"{self.problem['solution']}"
        elif method == "Second-Stage_Reasoning":
            return f"The answer is {self.answer_text}."
        elif method == "without_R":
            return f"The answer is {self.answer_text}."
        else:
            raise ValueError("Please Check the Value of Method.")

    def get_input(self, method):
        if method == "Explanation":
            return f"Question: {self.question_text}\nAnswer:"
        elif method == "Reasoning":
            return f"Question: {self.question_text}\nSolution:"
        elif method == "First-Stage_Reasoning":
            return f"Question: {self.question_text}\nSolution:"
        elif method == "Second-Stage_Reasoning":
            return f"Question: {self.question_text}\nSolution: {self.solution_text}\nAnswer:"
        else:
            raise ValueError("Please Check the Value of Method.")

    def get_img(self, _dataset):
        if _dataset == "rad":
            return self.problem['image'][:-4]
        elif _dataset == "slake":
            return str(self.problem["img_id"])
        else:
            raise ValueError(f"Invalid _dataset value: {_dataset}. The value must be 'rad' or 'slake'.")

