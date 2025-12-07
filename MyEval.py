from MyDataset import VqaSlakeDataset, VqaRadDataset, SegVqaRadDataset
from MyModel import T5ForMultimodalGeneration
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import numpy as np
import argparse
import torch
import json

def eval_loop(_args):
    torch.manual_seed(_args.seed)  # pytorch random seed
    np.random.seed(_args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    img_shape = {
        "detr": (100, 256),
        "vit": (105, 1024)
    }
    pretrained_model_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/"
    pretrained_model_path = pretrained_model_path+"Rational/" if _args.rational else pretrained_model_path+"Answer/"
    if _args.no_validate:
        pretrained_model_path = pretrained_model_path+f"checkpoint-{_args.model_path}"
    # if _args.rational:
    #     if _args.no_validate:
    #         pretrained_model_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/Rational/checkpoint-{_args.model_path}"
    #     else:
    #         pretrained_model_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/Rational"
    # else:
    #     if _args.no_validate:
    #         pretrained_model_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/Answer/checkpoint-{_args.model_path}"
    #     else:
    #         pretrained_model_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/Answer"

    model = T5ForMultimodalGeneration.from_pretrained(pretrained_model_path, img_shape[_args.img_type])
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
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

    if _args.dataset_type == "slake":
        if _args.rational:
            if _args.caption:
                _text_file_path = f"/path/to/VQA-SLAKE/cap_{_args.text_file_path}.json"
                if _args.augment and _args.text_file_path == 'train':
                    _text_file_path = f"/path/to/VQA-SLAKE/aug_cap_train.json"
            else:
                _text_file_path = f"/path/to/VQA-SLAKE/{_args.text_file_path}.json"
        else:
            _text_file_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/Rational/{_args.text_file_path}.json"

        data_set = VqaSlakeDataset(
            _tokenizer=tokenizer,
            _text_file_path=_text_file_path,
            _img_file_path=_args.img_file_path,
            _img_name_map=_args.img_name_map,
            _rational=_args.rational,
            _caption=_args.caption,
            source_len=_args.source_len,
            target_len=_args.target_len
        )
    elif _args.dataset_type == "rad":
        if _args.rational:
            if _args.caption:
                _text_file_path = f"/path/to/VQA-RAD/cap_{_args.text_file_path}set.json"
                if _args.augment and _args.text_file_path == 'train':
                    _text_file_path = f"/path/to/VQA-RAD/aug_cap_trainset.json"
            else:
                _text_file_path = f"/path/to/VQA-RAD/{_args.text_file_path}set.json"
        else:
            _text_file_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/Rational/{_args.text_file_path}.json"

        if _args.seg:
            data_set = SegVqaRadDataset(
                _tokenizer=tokenizer,
                _text_file_path=_text_file_path,
                _img_file_path=_args.img_file_path,
                _img_name_map=_args.img_name_map,
                _rational=_args.rational,
                _caption=_args.caption,
                source_len=_args.source_len,
                target_len=_args.target_len
            )
        else:
            data_set = VqaRadDataset(
                _tokenizer=tokenizer,
                _text_file_path=_text_file_path,
                _img_file_path=_args.img_file_path,
                _img_name_map=_args.img_name_map,
                _rational=_args.rational,
                _caption=_args.caption,
                source_len=_args.source_len,
                target_len=_args.target_len
            )
    else:
        raise ValueError

    predictions = trainer.predict(test_dataset=data_set, max_length=256)
    preds, targets = predictions.predictions, predictions.label_ids

    # Replace -100 in the Preds/Targets as we can't decode them.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    targets = np.where(targets != -100, targets, tokenizer.pad_token_id)

    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    targets_text = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    preds_text = [test_pred_text.strip() for test_pred_text in preds_text]
    targets_text = [test_target_text.strip() for test_target_text in targets_text]

    # print(preds_text)
    # print(targets_text)
    problem_ids = data_set.problem_id

    if _args.rational:
        save_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/Rational/{_args.text_file_path}_solution.json"
    else:
        save_path = f"{_args.output_dir}{_args.img_type}_{_args.suffix}/Answer/{_args.text_file_path}_answer.json"
    questions_dict = {f"question_{problem_id}": preds_text[index] for index, problem_id in enumerate(problem_ids)}
    with open(save_path, "w", encoding="utf-8") as Output_File:
        json.dump(questions_dict, Output_File, ensure_ascii=False, indent=4)

def load_rational(_text_file_path, _rational_path, _text_output_file_path):
    # Load the content of rational.json
    with open(_rational_path, "r", encoding="utf-8") as Rational_File:
        rational = json.load(Rational_File)

    # Load the content of testset.json
    with open(_text_file_path, "r", encoding="utf-8") as Test_File:
        test_text_file = json.load(Test_File)

    # Updata testset with the values from rational
    for key, value in rational.items():
        question_number = key.split('_')[1]
        if question_number in test_text_file:
            test_text_file[question_number]['solution']=value
        else:
            raise ValueError

    # Save the updated testset
    with open(_text_output_file_path, "w", encoding="utf-8") as Test_Output_File:
        json.dump(test_text_file, Test_Output_File, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file_path', type=str, default='test', choices=['train', 'validate', 'test'])
    parser.add_argument('--img_file_path', type=str, default='/path/to/VQA-SLAKE/detr.pth')
    parser.add_argument('--img_name_map', type=str, default='/path/to/VQA-SLAKE/name_map.json')
    parser.add_argument('--model_path', type=str, default='None')
    parser.add_argument('--output_dir', type=str, default='./experiments/')
    parser.add_argument('--img_type', type=str, default='detr', choices=['detr', 'vit'])
    parser.add_argument('--source_len', type=int, default=512)
    parser.add_argument('--target_len', type=int, default=256)
    parser.add_argument('--eval_bs', type=int, default=2, help='Evaluation Batch Size')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--suffix', type=str, default="2", help='MyExperiments Suffix')
    parser.add_argument('--seg', action='store_true', default=False)
    parser.add_argument('--dataset_type', type=str, default='slake', choices=['rad', 'slake'])
    parser.add_argument('--caption', action='store_true', default=False)
    parser.add_argument('--rational', action='store_true', default=False)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--no_validate', action='store_false', default=True)
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    eval_loop(args)
    if args.rational:
        if args.dataset_type == 'slake':
            if args.caption:
                text_file_path = f"/path/to/VQA-SLAKE/cap_{args.text_file_path}.json"
                if args.augment and args.text_file_path == 'train':
                    text_file_path = f"/path/to/VQA-SLAKE/aug_cap_train.json"
            else:
                text_file_path = f"/path/to/VQA-SLAKE/{args.text_file_path}.json"
            load_rational(
                _text_file_path=text_file_path,
                _rational_path=f"{args.output_dir}{args.img_type}_{args.suffix}/Rational/{args.text_file_path}_solution.json",
                _text_output_file_path=f"{args.output_dir}{args.img_type}_{args.suffix}/Rational/{args.text_file_path}.json"
            )
        elif args.dataset_type == 'rad':
            if args.caption:
                text_file_path = f"/path/to/VQA-RAD/cap_{args.text_file_path}set.json"
                if args.augment and args.text_file_path == 'train':
                    text_file_path = f"/path/to/VQA-RAD/aug_cap_trainset.json"
            else:
                text_file_path = f"/path/to/VQA-RAD/{args.text_file_path}set.json"
            load_rational(
                _text_file_path=text_file_path,
                _rational_path=f"{args.output_dir}{args.img_type}_{args.suffix}/Rational/{args.text_file_path}_solution.json",
                _text_output_file_path=f"{args.output_dir}{args.img_type}_{args.suffix}/Rational/{args.text_file_path}.json"
            )
        else:
            raise ValueError















