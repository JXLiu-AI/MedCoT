import nltk
import evaluate
import argparse
import re
import os
import numpy as np
import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataset import ClosedMedVQADataset
from model import T5ForMultimodalGeneration

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
            metric_for_best_model="rougeL" if _args.method == "First-Stage_Reasoning" else "accuracy",
            predict_with_generate=True,
            generation_max_length=_args.target_len,
            load_best_model_at_end=False,
            report_to=["none"],
        )

    # ========== Define compute_metrics functions ==============================
    def postprocess_text(_preds, _labels):
        _preds = [pred.strip() for pred in _preds]
        _labels = [label.strip() for label in _labels]
        _preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in _preds]
        _labels = ["\n".join(nltk.sent_tokenize(label)) for label in _labels]
        return _preds, _labels

    def extract_ans(_ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(_ans)

        if len(res) == 1:
            _answer = res[0]  # 'A', 'B', ...
        else:
            _answer = "FAILED"
        return _answer

    def compute_metrics_rougel(eval_preds):
        metric = evaluate.load("./rouge.py")
        preds, targets = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the Preds/Targets as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        targets = np.where(targets != -100, targets, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_toekn_len"] = np.mean(prediction_lens)
        return result

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

    train_set = ClosedMedVQADataset(
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
        compute_metrics=compute_metrics_rougel if _args.rational else compute_metrics_acc
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
    parser.add_argument('--method', type=str, choices=["Explanation", "Reasoning", "First-Stage_Reasoning", "Second-Stage_Reasoning", "without_R"])
    parser.add_argument('--source_len', type=int, default=512)
    parser.add_argument('--target_len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--bs', type=int, default=8, help='Batch Size')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight Decay')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--dataset', type=str, choices=['rad', 'slake'])
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    train_loop(args)




