# MedCoT: Medical Chain-of-Thought

This repository contains the code for the paper **"MedCoT: Medical Chain of Thought via Hierarchical Expert"** (EMNLP 2024).

## Introduction

MedCoT is a novel framework for Medical Visual Question Answering (Med-VQA) that leverages the Chain-of-Thought (CoT) reasoning paradigm. It addresses the limitations of existing methods in handling complex medical reasoning tasks by decomposing them into intermediate rationales.

Key features:
- **Sparse Mixture of Experts (MoE)**: Integrates a `TopKSparseMoELayer` to adaptively fuse visual and textual features, allowing the model to specialize in different types of medical questions.
- **Two-Stage Reasoning**: Generates rationales first, then infers the final answer based on the generated rationale and the original input.
- **Unified Architecture**: Built upon the UnifiedQA-T5 backbone.

## File Structure

### Core Scripts
- `MyModel.py`: Defines the model architecture, including `SparseExpert`, `TopKSparseMoELayer`, and `JointEncoder`.
- `MyTrain.py`: The main training script for the MedCoT pipeline (both rationale generation and answer inference stages).
- `MyEval.py`: The evaluation script for generating rationales and answers using trained models.
- `Paper_Experiments.sh`: Contains the exact commands used to run the experiments reported in the paper.
- `MyDataset.py`: Handles data loading for VQA-RAD and VQA-SLAKE datasets.

### Auxiliary & Baseline Scripts
- `DirectAnswer.sh` / `DirectAnswer.py`: Scripts for training a baseline model that predicts the answer directly (or Rationale+Answer jointly) without the two-stage pipeline.
- `OpenEnd.sh` / `OpenEndTrain.py`: Scripts specifically designed for Open-Ended VQA tasks, supporting different input/output formats (e.g., Answer First, Rationale First).
- `extract_feature.sh` / `extract_feature.py`: Utility scripts to extract visual features (ViT or DETR) from raw images, which are then used as input for the training scripts.

## Usage

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Timm (for feature extraction)

### 1. Feature Extraction

Before training, you need to extract visual features from your dataset images.

```bash
# Extract DETR features
python extract_feature.py \
    --device cuda:0 \
    --img_type detr \
    --image_dir /path/to/images/ \
    --output_dir /path/to/output/
```
See `extract_feature.sh` for more examples.

### 2. Core Experiments (MedCoT)

The main training process consists of two stages: Rationale Generation and Answer Inference. You can find the full workflow in `Paper_Experiments.sh`.


#### Step 1: Generate Rationales
Use the trained model to generate rationales for the training and test sets. This step involves the Initial Diagnosis Specialist (`Initial_Specialist.py`) and the Follow-up Specialist (`Follow_up_Specialist.py`).

```bash
# Initial Diagnosis Specialist
python Initial_Specialist.py ...

# Follow-up Specialist
python Follow_up_Specialist.py ...
```

#### Step 2: Train Answer Inference Model
Trains the model to predict the final answer using the question, image, and the generated rationale.

```bash
python MyTrain.py \
    --dataset_type rad \
    --suffix 0 \
    --epoch 20 \
    --lr 5e-5 \
    --bs 8 \
    --eval_bs 8 \
    --source_len 512 \
    --target_len 32 \
    --train_text_file_path ./experiments/detr_0/Rational/train.json \
    --test_text_file_path ./experiments/detr_0/Rational/test.json \
    --img_file_path /path/to/detr.pth \
    --img_name_map /path/to/name_map.json \
    --pretrained_model_path /path/to/unifiedqa-t5-base \
    --output_dir ./experiments/ \
    --img_type detr
```

#### Step 3: Evaluate Answer Inference Model
Evaluate the trained model on the test set.

```bash
python MyEval.py \
    --dataset_type rad \
    --suffix 0 \
    --no_validate \
    --source_len 512 \
    --target_len 32 \
    --eval_bs 8 \
    --text_file_path test \
    --img_file_path /path/to/detr.pth \
    --img_name_map /path/to/name_map.json \
    --output_dir ./experiments/ \
    --img_type detr
```

### 3. Additional Experiments

#### Direct Answer Baseline
To train a model that directly answers questions (or generates Rationale+Answer in a single step) without the two-stage pipeline:

```bash
# Train Direct Answer (No Caption)
python DirectAnswer.py --dataset_type rad --suffix 2 ...

# Train Rationale + Answer (Jointly)
python DirectAnswer.py --dataset_type rad --rational --suffix 3 ...
```
Refer to `DirectAnswer.sh` for various configurations (with/without captions, different suffixes).

#### Open-Ended VQA
For open-ended question answering tasks:

```bash
# Train Open-Ended Model
python OpenEndTrain.py --dataset_type rad --answer_first --rational ...
```
Refer to `OpenEnd.sh` for configurations like `--answer_first` (predict answer then rationale) or standard order.
