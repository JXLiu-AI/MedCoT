import os
from dataset import OpenMedVQADataset
from model import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import argparse
import torch
import json

def eval_loop(_args):
    torch.manual_seed(_args.seed)  # pytorch random seed
    np.random.seed(_args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    model = T5ForMultimodalGeneration.from_pretrained(_args.pretrained_model_path, (100, 256))
    tokenizer = AutoTokenizer.from_pretrained(_args.pretrained_model_path)
    datacollator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    config = Seq2SeqTrainingArguments(
        output_dir="./",
        per_device_eval_batch_size=_args.eval_bs,
        predict_with_generate=True,
        generation_max_length=_args.target_len,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        data_collator=datacollator
    )

    data_set = OpenMedVQADataset(
        _tokenizer=tokenizer,
        _text_file_path=_args.text_file_path,
        _img_file_path=_args.img_file_path,
        _img_name_map=_args.img_name_map,
        _method=_args.method,
        _source_len=_args.source_len,
        _target_len=_args.target_len,
        _dataset=_args.dataset
    )

    predictions = trainer.predict(test_dataset=data_set, max_length=256)
    preds, targets = predictions.predictions, predictions.label_ids

    # Replace -100 in the Preds/Targets as we can't decode them.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds_text = [test_pred_text.strip() for test_pred_text in preds_text]

    problem_ids = data_set.problem_id
    questions_dict = {f"question_{problem_id}": preds_text[index] for index, problem_id in enumerate(problem_ids)}
    if _args.method == "First-Stage_Reasoning":
        with open(_args.text_file_path, "r", encoding="utf-8") as RawFile:
            raw_data = json.load(RawFile)

        for key, value in questions_dict.items():
            question_number = key.split('_')[1]
            if question_number in raw_data:
                raw_data[question_number]['solution']=value
            else:
                raise ValueError
        if "test" in _args.text_file_path:
            save_path = os.path.join(_args.output_dir, _args.method, "test.json")
        else:
            save_path = os.path.join(_args.output_dir, _args.method, "train.json")
        with open(save_path, "w", encoding="utf-8") as OutputFile:
            json.dump(raw_data, OutputFile, ensure_ascii=False, indent=4)
    else:
        save_path = os.path.join(_args.output_dir, _args.method, "test_response.json")
        with open(save_path, "w", encoding="utf-8") as OutputFile:
            json.dump(questions_dict, OutputFile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file_path', type=str, default='None')
    parser.add_argument('--img_file_path', type=str, default='None')
    parser.add_argument('--img_name_map', type=str, default='None')
    parser.add_argument('--model_path', type=str, default='None')
    parser.add_argument('--output_dir', type=str, default='None')
    parser.add_argument('--source_len', type=int, default=512)
    parser.add_argument('--target_len', type=int, default=256)
    parser.add_argument('--eval_bs', type=int, default=8, help='Evaluation Batch Size')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--dataset', type=str, choices=['rad', 'slake'])
    parser.add_argument('--method', type=str, choices={"Explanation", "Reasoning", "First-Stage_Reasoning", "Second-Stage_Reasoning", "without_R"})
    parser.add_argument('--pretrained_model_path', type=str, default='None')
    parser.add_argument('--no_validate', action='store_false', default=True)
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    eval_loop(args)

