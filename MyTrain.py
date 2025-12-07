import nltk
import evaluate
import argparse
import re
import os
import json
import numpy as np
import torch
from MyDataset import VqaRadDataset, SegVqaRadDataset, VqaSlakeDataset, DirectAnswerDataset
from MyModel import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq


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

    root = _args.output_dir+f"{_args.img_type}_"+_args.suffix+"/"
    if not os.path.exists(root):
        os.mkdir(root)
    if _args.rational:
        save_dir = root+"Rational/"
    else:
        save_dir = root+"Answer/"
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
            metric_for_best_model="rougeL" if _args.rational else "accuracy",
            predict_with_generate=True,
            generation_max_length=_args.target_len,
            load_best_model_at_end=True if _args.no_validate else False,
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

    if _args.dataset_type == 'rad':
        if _args.seg:
            train_set = SegVqaRadDataset(
                _tokenizer=tokenizer,
                _text_file_path=_args.train_text_file_path,
                _img_file_path=_args.img_file_path,
                _img_name_map=_args.img_name_map,
                _rational=_args.rational,
                _caption=_args.caption,
                source_len=_args.source_len,
                target_len=_args.target_len
            )
            test_set = SegVqaRadDataset(
                _tokenizer=tokenizer,
                _text_file_path=_args.test_text_file_path,
                _img_file_path=_args.img_file_path,
                _img_name_map=_args.img_name_map,
                _rational=_args.rational,
                _caption=_args.caption,
                source_len=_args.source_len,
                target_len=_args.target_len
            )
        else:
            train_set = VqaRadDataset(
                _tokenizer=tokenizer,
                _text_file_path=_args.train_text_file_path,
                _img_file_path=_args.img_file_path,
                _img_name_map=_args.img_name_map,
                _rational=_args.rational,
                _caption=_args.caption,
                source_len=_args.source_len,
                target_len=_args.target_len
            )
            test_set = VqaRadDataset(
                _tokenizer=tokenizer,
                _text_file_path=_args.test_text_file_path,
                _img_file_path=_args.img_file_path,
                _img_name_map=_args.img_name_map,
                _rational=_args.rational,
                _caption=_args.caption,
                source_len=_args.source_len,
                target_len=_args.target_len
            )
        if _args.no_validate:
            trainer = Seq2SeqTrainer(
                model=model,
                args=config,
                train_dataset=train_set,
                eval_dataset=test_set,
                data_collator=datacollator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_rougel if _args.rational else compute_metrics_acc
            )
        else:
            trainer = Seq2SeqTrainer(
                model=model,
                args=config,
                train_dataset=train_set,
                data_collator=datacollator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_rougel if _args.rational else compute_metrics_acc
            )

    elif _args.dataset_type == 'slake':
        train_set = VqaSlakeDataset(
            _tokenizer=tokenizer,
            _text_file_path=_args.train_text_file_path,
            _img_file_path=_args.img_file_path,
            _img_name_map=_args.img_name_map,
            _rational=_args.rational,
            _caption=_args.caption,
            source_len=_args.source_len,
            target_len=_args.target_len
        )

        if _args.no_validate:
            # validate_set = VqaSlakeDataset(
            #     _tokenizer=tokenizer,
            #     _text_file_path=_args.validate_text_file_path,
            #     _img_file_path=_args.img_file_path,
            #     _img_name_map=_args.img_name_map,
            #     _rational=_args.rational,
            #     _caption=_args.caption,
            #     source_len=_args.source_len,
            #     target_len=_args.target_len
            # )
            test_set = VqaSlakeDataset(
                _tokenizer=tokenizer,
                _text_file_path=_args.test_text_file_path,
                _img_file_path=_args.img_file_path,
                _img_name_map=_args.img_name_map,
                _rational=_args.rational,
                _caption=_args.caption,
                source_len=_args.source_len,
                target_len=_args.target_len
            )
            # trainer = Seq2SeqTrainer(
            #     model=model,
            #     args=config,
            #     train_dataset=train_set,
            #     eval_dataset=validate_set,
            #     data_collator=datacollator,
            #     tokenizer=tokenizer,
            #     compute_metrics=compute_metrics_rougel if _args.rational else compute_metrics_acc
            # )
            trainer = Seq2SeqTrainer(
                model=model,
                args=config,
                train_dataset=train_set,
                eval_dataset=test_set,
                data_collator=datacollator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_rougel if _args.rational else compute_metrics_acc
            )
        else:
            trainer = Seq2SeqTrainer(
                model=model,
                args=config,
                train_dataset=train_set,
                data_collator=datacollator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_rougel if _args.rational else compute_metrics_acc
            )
    elif _args.dataset_type == 'rad2019':
        # import pdb;pdb.set_trace()
        train_set = DirectAnswerDataset(
            _tokenizer=tokenizer,
            _text_file_path=_args.train_text_file_path,
            _img_file_path=_args.img_file_path,
            _img_name_map=_args.img_name_map,
            _rational=_args.rational,
            _caption=_args.caption,
            source_len=_args.source_len,
            target_len=_args.target_len
        )
        test_set = DirectAnswerDataset(
            _tokenizer=tokenizer,
            _text_file_path=_args.test_text_file_path,
            _img_file_path=_args.img_file_path,
            _img_name_map=_args.img_name_map,
            _rational=_args.rational,
            _caption=_args.caption,
            source_len=_args.source_len,
            target_len=_args.target_len
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=config,
            train_dataset=train_set,
            eval_dataset=test_set,
            data_collator=datacollator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_rougel if _args.rational else compute_metrics_acc
        )
    else:
        raise ValueError(f"Invalid dataset value: {_args.dataset_type}. The value must be 'rad' or 'slake'.")

    trainer.train()
    trainer.save_model(save_dir)

    if _args.no_validate:
        metrics = trainer.evaluate(eval_dataset=test_set, max_length=_args.target_len)
        # print(f"Test_set Metrics is\n{metrics}")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # test_predictions = trainer.predict(test_dataset=test_set, max_length=_args.target_len)
    # test_preds, test_targets = test_predictions.predictions, test_predictions.label_ids
    #
    # # Replace -100 in the Preds/Targets as we can't decode them.
    # test_preds = np.where(test_preds != -100, test_preds, tokenizer.pad_token_id)
    #
    # test_preds_text = tokenizer.batch_decode(test_preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # test_preds_text = [test_pred_text.strip() for test_pred_text in test_preds_text]
    # test_problem_ids = test_set.problem_id
    #
    # if _args.rational:
    #     _save_path = save_dir+"testset_solution.json"
    # else:
    #     _save_path = save_dir+"test_answer.json"
    # save_test_context(test_problem_ids, test_preds_text, _save_path)


def save_test_context(_problem_ids: list, _text: list, _save_path):
    questions_dict = {f"question_{problem_id}": _text[index] for index, problem_id in enumerate(_problem_ids)}
    with open(_save_path, "w", encoding="utf-8") as Output_File:
        json.dump(questions_dict, Output_File, ensure_ascii=False, indent=4)

def load_test_rational(_test_text_file_path, _rational_path, _test_text_output_file_path):
    # Load the content of rational.json
    with open(_rational_path, "r", encoding="utf-8") as Rational_File:
        rational = json.load(Rational_File)

    # Load the content of testset.json
    with open(_test_text_file_path, "r", encoding="utf-8") as Test_File:
        test_text_file = json.load(Test_File)

    # Updata testset with the values from rational
    for key, value in rational.items():
        question_number = key.split('_')[1]
        if question_number in test_text_file:
            test_text_file[question_number]['solution']=value
        else:
            raise ValueError

    # Save the updated testset
    with open(_test_text_output_file_path, "w", encoding="utf-8") as Test_Output_File:
        json.dump(test_text_file, Test_Output_File, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_text_file_path', type=str, default='/path/to/VQA-RAD/trainset.json')
    parser.add_argument('--validate_text_file_path', type=str, default=None)
    parser.add_argument('--test_text_file_path', type=str, default='/path/to/VQA-RAD/testset.json')
    parser.add_argument('--img_file_path', type=str, default='/path/to/VQA-RAD/detr.pth')
    parser.add_argument('--img_name_map', type=str, default='/path/to/VQA-RAD/name_map.json')
    parser.add_argument('--pretrained_model_path', type=str, default='/path/to/unifiedqa-t5-base')
    parser.add_argument('--output_dir', type=str, default='./experiments/')
    parser.add_argument('--img_type', type=str, default='detr', choices=['detr', 'vit'])
    parser.add_argument('--rational', action='store_true', default=False)
    parser.add_argument('--caption', action='store_true', default=False)
    parser.add_argument('--no_validate', action='store_false', default=True)
    parser.add_argument('--source_len', type=int, default=512)
    parser.add_argument('--target_len', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--bs', type=int, default=4, help='Batch Size')
    parser.add_argument('--eval_bs', type=int, default=4, help='Evaluation Batch Size')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight Decay')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--suffix', type=str, default="zero", help='MyExperiments Suffix')
    parser.add_argument('--seg', action='store_true', default=False)
    parser.add_argument('--dataset_type', type=str, choices=['rad', 'slake', 'rad2019'])
    args = parser.parse_args()

    if args.dataset_type == "rad":
        if args.rational:
            for arg, value in vars(args).items():
                print(f"{arg}: {value}")
            train_loop(args)
        else:
            # test_text_file_path=args.test_text_file_path
            # rational_path = args.output_dir+f"{args.img_type}_{args.suffix}"+"/Rational/testset_solution.json"
            # test_text_output_file_path = args.output_dir+f"{args.img_type}_{args.suffix}"+"/Rational/testset.json"
            # load_test_rational(test_text_file_path, rational_path, test_text_output_file_path)
            # args.test_text_file_path = test_text_output_file_path
            for arg, value in vars(args).items():
                print(f"{arg}: {value}")
            train_loop(args)
    elif args.dataset_type == "slake":
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        train_loop(args)
    elif args.dataset_type == "rad2019":
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        train_loop(args)
    else:
        raise ValueError(f"Invalid dataset value: {args.dataset_type}. The value must be 'rad' or 'slake'.")









