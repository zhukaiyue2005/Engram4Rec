<div align=center>

<h1>Engram4Rec</h1>

<img src="https://img.shields.io/badge/License-MIT-blue" alt="license">

Engram4Rec is a demo repository for language-model-based sequential recommendation. It contains data construction scripts, supervised fine-tuning baselines, Engram-augmented recommenders, SASRec collaborative Engram models, and Softmax Direct Preference Optimization (Softmax-DPO) for preference optimization.

</div>

<p id="Catalogue"></p>

## 📋 Catalogue

- [Catalogue](#Catalogue)
- [Repository Structure](#Repository-Structure)
- [Preparations](#Preparations)
- [Data](#Data)
- [Quick Start](#Quick-Start)
- [Softmax-DPO](#Softmax-DPO)

<p id="Repository-Structure"></p>

## 📁 Repository Structure

```text
Engram4Rec/
├── data/
│   ├── Amazon/
│   │   ├── info/                         # item metadata files
│   │   ├── train/                        # raw train interaction files
│   │   ├── valid/                        # raw validation interaction files
│   │   └── test/                         # raw test interaction files
│   ├── Industrial_and_Scientific_dataset/ # jsonl prompts for LM training/evaluation
│   ├── process.py
│   └── build_data/
│       ├── build_industrial_data.py
│       └── build_industrial_data.sh
├── without_engram/                       # LM recommender without Engram memory
│   ├── sft.py
│   ├── sft.sh
│   ├── inference.py
│   ├── inference_SFT.sh
│   ├── softmax_dpo.py
│   ├── softmax_dpo.sh
│   ├── softmax_dpo_trainer.py
│   ├── softmax_dpo_utils.py
│   ├── KL_evaluate/
│   └── each_layer_hidden_states_beam_search/
├── with_normal_engram/                   # LM recommender with Engram insertion
│   ├── Engram_Insert_code/
│   ├── sft.py
│   ├── sft.sh
│   ├── inference.py
│   ├── inference_SFT.sh
│   ├── gate_analysis/
│   ├── CKA_evaluate/
│   ├── KL_evaluate/
│   └── each_layer_hidden_states_beam_search/
├── with_item_engram/
│   ├── Engram_Insert_code/
│   ├── sft.py
│   ├── sft.sh
│   ├── inference.py
│   └── inference_SFT.sh
└── with_SASRec_collabarative_engram/     # SASRec collaborative Engram model and analysis tools
    ├── train_sasrec.py
    ├── train_sasrec.sh
    ├── evaluate_sasrec.py
    ├── evaluate_sasrec.sh
    ├── sasrec_model.py
    ├── data_utils.py
    └── replace_item_abalation_inf/
```

<p id="Preparations"></p>

## ⚙️ Preparations

Create a Python environment, install PyTorch for your CUDA version, and then install the remaining dependencies.

For CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

The scripts expect a local or Hugging Face-compatible Qwen3 causal language model. Set it with `BASE_MODEL` when running training or inference:

```bash
export BASE_MODEL="Qwen3-1.7B"
```

For multi-GPU training, set visible devices and the number of processes before launching scripts:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
```

<p id="Data"></p>

## 🗂️ Data

The demo data is stored under `data/`.

- `data/Amazon/` contains raw Amazon split files and item metadata.
- `data/Industrial_and_Scientific_dataset/` contains prompt-style `train.jsonl`, `valid.jsonl`, and `test.jsonl` files used by LM-based recommenders.

To rebuild the Industrial and Scientific prompt dataset from the raw files:

```bash
cd data/build_data
bash build_industrial_data.sh
```

The generated files will be written to:

```text
data/Industrial_and_Scientific_dataset/
```

<p id="Quick-Start"></p>

## ⌛️ Quick Start

### 1. Train the LM recommender without Engram

```bash
cd without_engram
bash sft.sh
```

The script reads data from `../data/Industrial_and_Scientific_dataset/` and writes checkpoints/logs under `without_engram/`.

### 2. Run inference for the non-Engram model

Set `CHECKPOINT_PATH` to the LoRA checkpoint produced by SFT, then run:

```bash
cd without_engram
export CHECKPOINT_PATH="./sft_checkpoint"
bash inference_SFT.sh
```

### 3. Train the Engram-augmented recommender

```bash
cd with_normal_engram
bash sft.sh
```

The Engram implementation is under `with_normal_engram/Engram_Insert_code/`.

### 4. Train the SASRec collaborative Engram model

```bash
cd with_SASRec_collabarative_engram
bash train_sasrec.sh
```

<p id="Softmax-DPO"></p>

## 🔥 Softmax-DPO

Softmax-DPO optimizes the LM recommender with one positive item and multiple sampled negative items. The implementation is under `without_engram/`:

- `softmax_dpo.py`: Softmax-DPO training entry.
- `softmax_dpo.sh`: runnable shell script with relative data paths.
- `softmax_dpo_trainer.py`: custom trainer.
- `softmax_dpo_utils.py`: data collator and trainer utilities.

Before running Softmax-DPO, prepare an SFT LoRA checkpoint and point `SFT_CHECKPOINT` to it:

```bash
cd without_engram
export BASE_MODEL="Qwen3-1.7B"
export SFT_CHECKPOINT="./sft_checkpoint"
bash softmax_dpo.sh
```

Useful environment variables:

```bash
export OUTPUT_DIR="./softmax_dpo_output"
export LOG_FILE="softmax_dpo.log"
export DPO_BETA=0.1
export NEG_NUM=3
export LEARNING_RATE=1e-5
export NUM_TRAIN_EPOCHS=1
```

All default data paths in `softmax_dpo.py` and `softmax_dpo.sh` are relative to this repository.
