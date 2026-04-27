<div align=center>

<h1>Engram4Rec</h1>

<img src="https://img.shields.io/badge/License-MIT-blue" alt="license">

Engram4Rec is a demo repository for language-model-based sequential recommendation. It contains data construction scripts, supervised fine-tuning baselines, Engram-augmented recommenders, SASRec collaborative Engram models, and Softmax Direct Preference Optimization (Softmax-DPO) for preference optimization.

</div>

## Recommendation Results

Qwen3-1.7B is used as the backbone model. It contains 28 transformer layers, and Engram modules are inserted at layers 7, 14, and 21. The best performance is highlighted in boldface, while the second-best performance is underlined. Since HR@1 and NDCG@1 are identical, they are reported as one combined metric.

[PDF version](result.pdf)

| Dataset | Model | HR/NDCG@1 | HR@3 | NDCG@3 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Toys | GRU4Rec | 0.0098 | 0.0181 | 0.0145 | 0.0233 | 0.0166 | 0.0342 | 0.0201 |
| Toys | Caser | 0.0105 | 0.0197 | 0.0158 | 0.0268 | 0.0187 | 0.0387 | 0.0225 |
| Toys | SASRec | 0.0206 | 0.0351 | 0.0290 | 0.0436 | 0.0325 | 0.0570 | 0.0369 |
| Toys | Without Engram | 0.0190 | 0.0385 | 0.0304 | 0.0537 | 0.0367 | 0.0797 | 0.0450 |
| Toys | Normal Engram | <u>0.0255</u> | <u>0.0435</u> | <u>0.0358</u> | 0.0565 | <u>0.0411</u> | 0.0802 | <u>0.0487</u> |
| Toys | Item-Engram | 0.0208 | 0.0415 | 0.0327 | <u>0.0567</u> | 0.0390 | <u>0.0805</u> | 0.0466 |
| Toys | SAS-Engram | **0.0293** | **0.0490** | **0.0407** | **0.0614** | **0.0458** | **0.0821** | **0.0524** |
| Toys | Cross-Attention | 0.0220 | 0.0424 | 0.0338 | 0.0553 | 0.0391 | 0.0780 | 0.0464 |
| Industrial | GRU4Rec | 0.0454 | 0.0664 | 0.0578 | 0.0801 | 0.0634 | 0.1006 | 0.0700 |
| Industrial | Caser | 0.0465 | 0.0633 | 0.0564 | 0.0735 | 0.0606 | 0.0944 | 0.0672 |
| Industrial | SASRec | 0.0545 | 0.0803 | 0.0699 | 0.0924 | 0.0749 | 0.1141 | 0.0817 |
| Industrial | Without Engram | 0.0711 | 0.1000 | 0.0878 | 0.1214 | 0.0966 | 0.1472 | 0.1050 |
| Industrial | Normal Engram | <u>0.0744</u> | <u>0.1079</u> | <u>0.0938</u> | 0.1249 | <u>0.1008</u> | **0.1487** | **0.1086** |
| Industrial | Item-Engram | **0.0763** | **0.1083** | **0.0951** | <u>0.1251</u> | **0.1019** | 0.1443 | <u>0.1081</u> |
| Industrial | SAS-Engram | 0.0684 | 0.1048 | 0.0894 | **0.1256** | 0.0979 | <u>0.1476</u> | 0.1051 |
| Industrial | Cross-Attention | 0.0649 | 0.0918 | 0.0807 | 0.1057 | 0.0864 | 0.1342 | 0.0956 |
| Office | GRU4Rec | 0.0423 | 0.0701 | 0.0586 | 0.0853 | 0.0648 | 0.1106 | 0.0730 |
| Office | Caser | 0.0417 | 0.0758 | 0.0613 | 0.0915 | 0.0677 | 0.1136 | 0.0748 |
| Office | SASRec | 0.0512 | 0.0841 | 0.0704 | 0.0978 | 0.0760 | 0.1200 | 0.0832 |
| Office | Without Engram | 0.0684 | 0.1200 | 0.0982 | 0.1564 | 0.1132 | 0.2203 | 0.1336 |
| Office | Normal Engram | **0.0980** | **0.1486** | **0.1271** | **0.1753** | **0.1380** | **0.2310** | **0.1559** |
| Office | Item-Engram | 0.0814 | <u>0.1354</u> | 0.1126 | <u>0.1702</u> | <u>0.1268</u> | 0.2195 | 0.1428 |
| Office | SAS-Engram | <u>0.0849</u> | 0.1352 | <u>0.1140</u> | 0.1640 | 0.1258 | 0.2199 | 0.1436 |
| Office | Cross-Attention | 0.0818 | 0.1328 | 0.1110 | 0.1663 | 0.1247 | <u>0.2281</u> | <u>0.1445</u> |

<p id="Catalogue"></p>

## 📋 Catalogue

- [Catalogue](#Catalogue)
- [Recommendation Results](#recommendation-results)
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
└── with_SASRec_collaborative_engram/     # SASRec collaborative Engram model and analysis tools
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

This variant uses a pretrained SASRec item embedding table. Train the SASRec model first:

```bash
cd with_SASRec_collaborative_engram
bash train_sasrec.sh
```

By default, `train_sasrec.sh` writes the best checkpoint to:

```text
with_SASRec_collaborative_engram/SAS-checkpoints/sasrec_best.pt
```

Then run LM fine-tuning with the SASRec checkpoint:

```bash
cd with_SASRec_collaborative_engram
export SASREC_CHECKPOINT_PATH="./SAS-checkpoints/sasrec_best.pt"
bash sft.sh
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
