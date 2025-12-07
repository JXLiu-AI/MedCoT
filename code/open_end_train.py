import torch
import argparse
import os
import numpy as np
from MyModel import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataset import OpenMedVQADataset

def train_loop(_args):
    torch.manual_seed(_args.seed)  # pytorch random seed
    np.random.seed(_args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    model = T5ForMultimodalGeneration.from_pretrained(_args.pretrained_model_path, (100, 256))
    tokenizer = AutoTokenizer.from_pretrained(_args.pretrained_model_path)
    datacollator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    save_dir = os.path.join(_args.output_dir, _args.method)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    config = Seq2SeqTrainingArguments(
            output_dir=save_dir,
            evaluation_strategy="no",
            logging_strategy="epoch",
            save_strategy="no",
            save_total_limit=1,
            learning_rate=_args.lr,
            per_device_train_batch_size=_args.bs,
            weight_decay=_args.wd,
            num_train_epochs=_args.epoch,
            predict_with_generate=True,
            generation_max_length=_args.target_len,
            load_best_model_at_end=False,
            report_to=["none"],
        )

    train_set = OpenMedVQADataset(
        _tokenizer=tokenizer,
        _text_file_path=_args.train_text_file_path,
        _img_file_path=_args.img_file_path,
        _img_name_map=_args.img_name_map,
        _method=_args.method,
        _source_len=_args.source_len,
        _target_len=_args.target_len,
        _dataset=_args.dataset
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=config,
        train_dataset=train_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_text_file_path', type=str, default='None')
    parser.add_argument('--img_file_path', type=str, default='None')
    parser.add_argument('--img_name_map', type=str, default='None')
    parser.add_argument('--pretrained_model_path', type=str, default='None')
    parser.add_argument('--output_dir', type=str, default='None')
    parser.add_argument('--method', type=str, choices={"Explanation", "Reasoning", "First-Stage_Reasoning", "Second-Stage_Reasoning", "without_R"})
    parser.add_argument('--source_len', type=int, default=512)
    parser.add_argument('--target_len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--bs', type=int, default=4, help='Batch Size')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight Decay')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--dataset', type=str, choices=['rad', 'slake'])
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    train_loop(args)

























