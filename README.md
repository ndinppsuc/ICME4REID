# GDS-ReID: Graph-Guided Dual-Stream Collaborative Reasoning for Text-Based Person Re-Identification

<div align="center">

<img src="page1.pngpng" width="80%">

[![Paper](https://img.shields.io/badge/Paper-ICME_Submission-blue)](ICME4ReID.pdf)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

## 📢 News
*  This paper has been submitted to **ICME 2025**.
* **[Important]** Pre-trained weights and checkpoints will be released **upon paper acceptance**.

## 📖 Introduction

This repository contains the official implementation of the paper **"GDS-ReID: Graph-Guided Dual-Stream Collaborative Reasoning for Text-Based Person Re-Identification"**.

**Abstract:**
In interactive text-based person re-identification (ReID), particularly in scenarios exhibiting high visual similarity between the target and candidates, fine-grained inter-candidate differences often serve as the primary basis for distinguishing the target from distractors. However, existing methods typically treat retrieved candidates as a static and mutually independent ranked list. 

To address this, we propose **GDS-ReID**, a graph-guided dual-stream collaborative reasoning framework.
* **Visual Affinity Graph:** Constructs a graph over the candidate set to expose local ambiguity via nearest-neighbor relations.
* **Dual-Stream Reasoning:**
    * **Global Stream:** Verifies attribute-level consistency to stabilize search direction.
    * **Local Stream:** Performs neighbor-aware pairwise comparison to extract fine-grained differential evidence using mined hard negatives.

<div align="center">
  <img src="fig.png" width="100%">
  <br>
  <em>Figure 1: The overall framework of GDS-ReID.</em>
</div>

## 📂 Project Structure

```text
.
├── config/                 # Configuration files (YAML/JSON)
├── data/                   # Data processing and prompt definitions
├── model/                  # Model definitions (LLaVA-ReID, IRRA, etc.)
├── reid_datasets/          # Dataset loading scripts
├── train_llava_reid.py     # Main training script
├── main_eval.py            # Evaluation script
├── requirements.txt        # Dependencies
└── train_question.sh       # Training launch script

```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone [https://github.com/YourUsername/GDS-ReID.git](https://github.com/YourUsername/GDS-ReID.git)
cd GDS-ReID

```


2. **Create a conda environment**
```bash
conda create -n gdsreid python=3.10
conda activate gdsreid

```


3. **Install dependencies**
```bash
pip install -r requirements.txt

```



## 🚀 How to Run

### 1. Data Preparation

Please organize your datasets (CUHK-PEDES, ICFG-PEDES, RSTPReid) in the `data/` directory.

### 2. Training

Our framework consists of two training stages:

**Step 1: Train the Questioner**

```bash
bash train_question.sh
# Or manually:
# deepspeed train_llava_reid.py --config config/train_question.yaml

```

**Step 2: Train the Selector**

```bash
deepspeed train_llava_reid.py --config config/train_selector.yaml

```

### 3. Evaluation

To evaluate the model on the test sets:

```bash
python main_eval.py --config config/Interactive-PEDES.yaml \
    --resume /path/to/your/checkpoint/pytorch_model.bin

```

## 🙏 Acknowledgements

We would like to express our gratitude to the following open-source projects, which provided the foundation for our work:

* **[LLaVA-ReID](https://github.com/XLearning-SCU/LLaVA-ReID)**
* **[IRRA](https://www.google.com/search?q=https://github.com/Ukcheung/IRRA)**: Cross-Modal Implicit Relation Reasoning and Alignment for Text-to-Image Person Retrieval.

