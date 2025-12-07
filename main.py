import json
import os

import numpy as np
import pandas as pd
from rich import box
from rich.table import Column, Table
from transformers import T5Tokenizer

from src import constants
from src.args_parser import parse_args
from src.data.fakeddit.dataset import FakedditDataset
from src.data.scienceQA.data import load_data
from src.runner.chain_of_thought import ChainOfThought
from src.models.t5_multimodal_generation.training_params import (
    get_t5_model, get_training_data)
from src.models.t5_multimodal_generation.utils import get_backup_dir
from dotenv import load_dotenv

from src.utils import parse_range


args = parse_args()

def get_fakeddit_cot():

    data_range_start, data_rage_end = parse_range(args.data_range)
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model)
    model = get_t5_model(args, tokenizer, get_backup_dir(args))

    dataframe = pd.read_csv(constants.FAKEDDIT_DATASET_PATH)

    rationales = None
    if args.test_le:
        with open(args.test_le, "r") as f:
            rationale_df = json.loads(f.read())
        rationales = rationale_df['predictions'][data_range_start:data_rage_end]

    vision_features = None

    if args.img_type == "detr_facebook":
        vision_features = np.load(
            constants.FAKEDDIT_VISION_FEATURES_DETR_FULL_PATH, allow_pickle=True)[data_range_start:data_rage_end]

    elif args.img_type == "cooelf_detr":
        vision_features = np.load(
            constants.FAKEDDIT_VISION_FEATURES_COOELF_DETR_FULL_PATH, allow_pickle=True)[data_range_start:data_rage_end]

    test_set = FakedditDataset(
        dataframe=dataframe[data_range_start:data_rage_end],
        tokenizer=tokenizer,
        vision_features=vision_features,
        rationales=rationales,
        prompt=args.prompt
    )
    chain_of_thought = ChainOfThought(args) \
        .set_tokenizer(tokenizer) \
        .set_eval_set(test_set) \
        .set_test_set(test_set) \
        .set_model(model)

    return chain_of_thought


def get_science_qa_cot():
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model)
    model = get_t5_model(args, tokenizer, get_backup_dir(args))

    problems, qids, name_maps, image_features = load_data(args)
    dataframe = {
        'problems': problems,
        'qids': qids,
        'name_maps': name_maps,
        'image_features': image_features
    }

    train_set, eval_set, test_set = get_training_data(
        args, dataframe, tokenizer)

    chain_of_thought = ChainOfThought(args) \
        .set_tokenizer(tokenizer) \
        .set_train_set(train_set) \
        .set_eval_set(eval_set) \
        .set_test_set(test_set) \
        .set_model(model)

    return chain_of_thought


if __name__ == '__main__':

    # import nltk
    # nltk.download('punkt')

    load_dotenv(override=True)

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    cot_map = {
        constants.DatasetType.FAKEDDIT.value: get_fakeddit_cot,
        constants.DatasetType.SCIENCEQA.value: get_science_qa_cot,
    }
    cot = cot_map.get(args.dataset)()
    cot.run()

print("OKAY")