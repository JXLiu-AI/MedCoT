import json
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
import evaluate
import argparse
import re
import os
import numpy as np
from MyModel import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

class OpenAnswerDataset(Dataset):
    def __init__(self, _tokenizer, _text_file_path, _img_file_path, _img_name_map,
                 _rational, _caption, _answer_first, _dataset, source_len, target_len):
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
            prompt = pair.get_input(_rational, _caption, _answer_first)
            target = pair.get_target(_rational, _answer_first)
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

class InputAndTargetAndImg:
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

    def get_target(self, _rational=True, _answer_first=False):
        if _rational:
            if _answer_first:
                return f"The answer is {self.answer_text}.\nSolution: {self.problem['solution']}"
            else:
                return f"{self.problem['solution']}\nAnswer: The answer is {self.answer_text}."
        else:
            return f"The answer is {self.answer_text}."

    def get_input(self, _rational=True, _caption=False, _answer_first=False):
        prior_text = f"Question: {self.question_text}\n"
        if _caption:
            prior_text = prior_text + f"Caption: {self.problem['caption']}\n"
        if _rational and not _answer_first:
            return prior_text + "Solution:"
        else:
            # return prior_text + "Answer:"
            return prior_text+f"Solution: {self.solution_text}\nAnswer:"

    def get_img(self, _dataset):
        if _dataset == "rad":
            return self.problem['image'][:-4]
        elif _dataset == "slake":
            return str(self.problem["img_id"])
        else:
            raise ValueError(f"Invalid _dataset value: {_dataset}. The value must be 'rad' or 'slake'.")

def save_test_context(_problem_ids: list, _text: list, _save_path):
    questions_dict = {f"question_{problem_id}": _text[index] for index, problem_id in enumerate(_problem_ids)}
    with open(_save_path, "w", encoding="utf-8") as Output_File:
        json.dump(questions_dict, Output_File, ensure_ascii=False, indent=4)

def train_loop(_args):
    torch.manual_seed(_args.seed)  # pytorch random seed
    np.random.seed(_args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    img_shape = {
        "detr": (100, 256),
        "vit": (105, 1024)
    }
    model = T5ForMultimodalGeneration.from_pretrained(_args.pretrained_model_path, img_shape[_args.img_type])
    tokenizer = AutoTokenizer.from_pretrained(_args.pretrained_model_path)
    datacollator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    save_dir = os.path.join(_args.output_dir, f"{_args.img_type}_{_args.suffix}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    config = Seq2SeqTrainingArguments(
            output_dir=save_dir,
            evaluation_strategy="epoch" if _args.no_validate else "no",
            logging_strategy="epoch",
            save_strategy="epoch" if _args.no_validate else "no",
            save_total_limit=2 if _args.no_validate else 1,
            learning_rate=_args.lr,
            per_device_train_batch_size=_args.bs,
            per_device_eval_batch_size=_args.eval_bs,
            weight_decay=_args.wd,
            num_train_epochs=_args.epoch,
            # metric_for_best_model="accuracy",
            predict_with_generate=True,
            generation_max_length=_args.target_len,
            load_best_model_at_end=True if _args.no_validate else False,
            report_to=["none"],
        )

    # ========== Define compute_metrics functions ==============================
    def extract_ans(_ans):
        pattern = re.compile(r'The answer is \(([A-B])\)')
        res = pattern.findall(_ans)

        if len(res) == 1:
            _answer = res[0]  # 'A', 'B', ...
        else:
            _answer = "FAILED"
        return _answer

    def compute_metrics_acc(eval_preds):
        preds, targets = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100 in the Preds/Targets as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        targets = np.where(targets != -100, targets, tokenizer.pad_token_id)

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct +=1
        return {'accuracy': 1.0*correct/len(targets)}
    # ========== Define compute_metrics functions ==============================
    train_set = OpenAnswerDataset(
        _tokenizer=tokenizer,
        _text_file_path=_args.train_text_file_path,
        _img_file_path=_args.img_file_path,
        _img_name_map=_args.img_name_map,
        _rational=_args.rational,
        _caption=_args.caption,
        _answer_first=_args.answer_first,
        source_len=_args.source_len,
        target_len=_args.target_len,
        _dataset=_args.dataset_type
    )
    test_set = OpenAnswerDataset(
        _tokenizer=tokenizer,
        _text_file_path=_args.test_text_file_path,
        _img_file_path=_args.img_file_path,
        _img_name_map=_args.img_name_map,
        _rational=_args.rational,
        _caption=_args.caption,
        _answer_first=_args.answer_first,
        source_len=_args.source_len,
        target_len=_args.target_len,
        _dataset=_args.dataset_type
    )
    if _args.no_validate:
        trainer = Seq2SeqTrainer(
            model=model,
            args=config,
            train_dataset=train_set,
            eval_dataset=test_set,
            data_collator=datacollator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_acc
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=config,
            train_dataset=train_set,
            data_collator=datacollator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_acc
        )

    trainer.train()
    trainer.save_model(save_dir)
    if _args.no_validate:
        metrics = trainer.evaluate(eval_dataset=test_set, max_length=_args.target_len)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    print("Save the output...")
    test_predictions = trainer.predict(test_dataset=test_set, max_length=_args.target_len)
    test_preds, test_targets = test_predictions.predictions, test_predictions.label_ids

    # Replace -100 in the Preds/Targets as we can't decode them.
    test_preds = np.where(test_preds != -100, test_preds, tokenizer.pad_token_id)

    test_preds_text = tokenizer.batch_decode(test_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    test_preds_text = [test_pred_text.strip() for test_pred_text in test_preds_text]
    test_problem_ids = test_set.problem_id
    save_result_path = os.path.join(save_dir, "test_response.json")
    save_test_context(test_problem_ids, test_preds_text, save_result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_text_file_path', type=str, default='/path/to/VQA-RAD/trainset.json')
    parser.add_argument('--test_text_file_path', type=str, default='/path/to/VQA-RAD/testset.json')
    parser.add_argument('--img_file_path', type=str, default='/path/to/VQA-RAD/detr.pth')
    parser.add_argument('--img_name_map', type=str, default='/path/to/VQA-RAD/name_map.json')
    parser.add_argument('--pretrained_model_path', type=str, default='/path/to/unifiedqa-t5-base')
    parser.add_argument('--output_dir', type=str, default='./experiments/')
    parser.add_argument('--img_type', type=str, default='detr', choices=['detr', 'vit'])
    parser.add_argument('--rational', action='store_true', default=False)
    parser.add_argument('--caption', action='store_true', default=False)
    parser.add_argument('--no_validate', action='store_false', default=True)
    parser.add_argument('--answer_first', action='store_true', default=False)
    parser.add_argument('--source_len', type=int, default=512)
    parser.add_argument('--target_len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--bs', type=int, default=4, help='Batch Size')
    parser.add_argument('--eval_bs', type=int, default=4, help='Evaluation Batch Size')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight Decay')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--suffix', type=str, default="zero", help='MyExperiments Suffix')
    parser.add_argument('--dataset_type', type=str, choices=['rad', 'slake'])
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    train_loop(args)

















