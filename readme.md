# PAN Voight-Kampff Generative AI Detection (Task 1 & Task 2)

This repository contains code for a reproducible research pipeline on **Generative AI authorship detection**, developed for the **PAN Voight-Kampff 2025 shared task**. The project covers:

- **Task 1**: Binary classification (human vs. machine-generated text)
- **Task 2**: Fine-grained 6-class human–AI collaboration patterns

The workflow spans classical baselines, neural models, stylometric features, hybrid architectures, robustness testing, transfer learning, and qualitative error analysis.

> **Data access notice**  
> The Task 1 and Task 2 datasets are obtained by application from:
> https://pan.webis.de/shared-tasks.html#generated-content-analysis  
> (Voight-Kampff Generative AI Detection, PAN 2025).  
> **The data cannot be redistributed or uploaded to GitHub.**

---

## Directory Structure
The project assumes a **simple, flat structure** to make experimentation convenient. All Task 1 scripts live in the same folder as the Task 1 data, while Task 2 data is kept separately.

```
Desktop/
├── task1/                         # Main working directory (all scripts live here)
│   ├── train.jsonl                # PAN Voight-Kampff Task 1 training split (binary labels)
│   ├── val.jsonl                  # PAN Voight-Kampff Task 1 validation split
│   │
│   ├── inspect_jsonl.py           # Quick sanity check of JSONL format and fields
│   ├── load_with_pandas.py        # Load JSONL with pandas (shape, columns, label counts)
│   ├── dataset_loader.py          # Unified dataset loader used across models
│   ├── eda_basic.py               # Exploratory data analysis (labels, lengths, generators)
│   │
│   ├── baseline_tfidf_logreg.py   # Classical TF–IDF + Logistic Regression baseline
│   ├── t1_bert_baseline.py        # RoBERTa-base fine-tuning baseline (Task 1)
│   │
│   ├── style_features.py          # Stylometric feature definitions
│   ├── t1_compute_style_features.py # Extract stylometric features and save to disk
│   ├── t1_hybrid_nocontrast.py    # Hybrid (RoBERTa + stylometry), no contrastive loss
│   ├── t1_hybrid_contrastive.py   # Hybrid model with contrastive learning objective
│   │
│   ├── t1_predict_roberta.py      # Generate prediction scores from trained RoBERTa
│   ├── t1_predict_roberta_demo.py # Lightweight demo version of prediction
│   ├── t1_paraphrase_attack_demo.py # Robustness test via paraphrasing attack
│   ├── t1_on_task2_binary.py      # Transfer Task 1 model to Task 2 (binary collapse)
│   ├── t1_error_samples.py        # Extract false positives / false negatives for analysis
│   │
│   ├── t2_inspect_labels.py       # Inspect and sanity-check Task 2 labels
│   ├── t2_roberta_6class.py       # RoBERTa-based 6-way classifier (Task 2)
│   ├── t2_results_6class_demo.py  # Summarize Task 2 6-class results
│   ├── t2_derive_binary_from_6class.py # Collapse 6-class predictions to binary
│   ├── t2_derive_binary_from_6class_demo.py # Demo version of binary derivation
│   ├── t2_sliding_window_demo.py  # Sliding-window inference for long documents
│   │
│   └── metrics.py                 # Shared evaluation utilities (AUC, F1, accuracy)
│
└── task2/
    ├── task2_train.jsonl           # PAN Voight-Kampff Task 2 training split (6-class)
    └── task2_dev.jsonl             # PAN Voight-Kampff Task 2 development split
```

---

## Environment Setup
Python **>= 3.10** is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Execution Order
The scripts are designed to be run **incrementally**, where earlier steps generate artifacts (statistics, features, models, predictions) that later steps depend on.

### Phase 0: Data Inspection & Sanity Checks
Purpose: verify data integrity before modeling.

```bash
python inspect_jsonl.py
python load_with_pandas.py
python dataset_loader.py
python eda_basic.py
```

---

### Phase 1: Task 1 Baselines
Purpose: establish strong reference baselines for binary AI vs. human detection.

```bash
python baseline_tfidf_logreg.py
python t1_bert_baseline.py
```

---

### Phase 2: Stylometry & Hybrid Models
Purpose: test whether explicit stylometric signals complement neural representations.

```bash
python t1_compute_style_features.py
python t1_hybrid_nocontrast.py
python t1_hybrid_contrastive.py
```

---

### Phase 3: Robustness & Transfer Analyses
Purpose: evaluate generalization under distribution shifts.

```bash
python t1_on_task2_binary.py
python t1_paraphrase_attack_demo.py
```

---

### Phase 4: Task 2 (6-Class Authorship Patterns)
Purpose: model fine-grained human–AI collaboration categories.

```bash
python t2_inspect_labels.py
python t2_roberta_6class.py
python t2_derive_binary_from_6class.py
```

---

### Phase 5: Analysis & Error Inspection
Purpose: qualitative and diagnostic analysis for the paper.

```bash
python t2_sliding_window_demo.py
python t1_error_samples.py
```

---

This execution order follows the experimental flow of the paper.

Phases 0–1 cover data inspection and baseline construction, establishing the binary Task 1 setting and its lexical and Transformer-based reference models. Phases 2–3 introduce the hybrid architectures and robustness experiments, including cross-task transfer and paraphrasing-based perturbations, which constitute the main empirical analyses. Phase 4 extends the framework to the six-class Task 2 collaboration setting and its corresponding binary projections, while Phase 5 focuses on qualitative analysis through sliding-window inference and targeted error inspection.

