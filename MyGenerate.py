from MyDataset import VqaSlakeDataset, VqaRadDataset, SegVqaRadDataset
from MyModel import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import argparse
import torch
import json
import os


def eval_loop(_args):
    torch.manual_seed(_args.seed)  # pytorch random seed
    np.random.seed(_args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    model = T5ForMultimodalGeneration.from_pretrained(_args.model_path, (100, 256))
    tokenizer = AutoTokenizer.from_pretrained(_args.model_path)
    datacollator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    config = Seq2SeqTrainingArguments(
        output_dir="./",
        per_device_eval_batch_size=32,
        predict_with_generate=True,
        generation_max_length=_args.target_len,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        data_collator=datacollator
    )

    if _args.dataset_type == "slake":
        data_set = VqaSlakeDataset(
            _tokenizer=tokenizer,
            _text_file_path=_args.text_file_path,
            _img_file_path=_args.img_file_path,
            _img_name_map=_args.img_name_map,
            _rational=_args.rationale,
            _caption=_args.caption,
            source_len=_args.source_len,
            target_len=_args.target_len
        )
    elif _args.dataset_type == "rad":
        data_set = VqaRadDataset(
            _tokenizer=tokenizer,
            _text_file_path=_args.text_file_path,
            _img_file_path=_args.img_file_path,
            _img_name_map=_args.img_name_map,
            _rational=_args.rationale,
            _caption=_args.caption,
            source_len=_args.source_len,
            target_len=_args.target_len
        )

    predictions = trainer.predict(test_dataset=data_set, max_length=256)
    preds, targets = predictions.predictions, predictions.label_ids

    # Replace -100 in the Preds/Targets as we can't decode them.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    preds_text = [test_pred_text.strip() for test_pred_text in preds_text]
    problem_ids = data_set.problem_id

    save_path = os.path.join('./experiments/Final_Result', _args.output_path)
    questions_dict = {f"question_{problem_id}": preds_text[index] for index, problem_id in enumerate(problem_ids)}
    with open(save_path, "w", encoding="utf-8") as Output_File:
        json.dump(questions_dict, Output_File, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file_path', type=str, default='/path/to/VQA-SLAKE/cap_test.json')
    parser.add_argument('--img_file_path', type=str, default='/path/to/VQA-SLAKE/detr.pth')
    parser.add_argument('--img_name_map', type=str, default='/path/to/VQA-SLAKE/name_map.json')
    parser.add_argument('--dataset_type', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='None')
    parser.add_argument('--output_path', type=str, default='None')
    parser.add_argument('--source_len', type=int, default=512)
    parser.add_argument('--target_len', type=int, default=256)
    parser.add_argument('--rationale', action='store_false', default=True)
    parser.add_argument('--caption', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    eval_loop(args)















