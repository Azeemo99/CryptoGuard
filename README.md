# CryptoGuard

AI-powered detection of social engineering threats in blockchain communications.

This project trains and evaluates eight machine learning models — spanning TF-IDF, DistilBERT, BERT, and RoBERTa — across two training regimes (general email corpus and blockchain-adapted) to detect phishing and social engineering in blockchain communication contexts. The core finding is a substantial and consistent domain gap between general phishing detection performance and blockchain-specific performance across all three transformer architectures, and that domain adaptation closes this gap robustly.

Built as a final year project for BSc Computer Science at the University of East Anglia, 2025–2026.

---

## Project Structure

```
cryptoguard/
├── data/
│   ├── general/              # General email corpus (train/val/test splits)
│   └── blockchain/           # Synthetic blockchain corpus + splits
├── models/                   # Trained model checkpoints (8 total)
│   ├── tfidf_general/
│   ├── tfidf_blockchain/
│   ├── distilbert_general/
│   ├── bert_general/
│   ├── roberta_general/
│   ├── distilbert_blockchain/
│   ├── bert_blockchain/
│   └── roberta_blockchain/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_distilbert_training.ipynb
│   ├── 05_blockchain_data_collection.ipynb
│   ├── 06_synthetic_data_cleaning.ipynb
│   ├── 07_blockchain_data_split.ipynb
│   ├── 08.00-Train-Models.ipynb       # Colab — trains all 8 models
│   ├── 09_evaluate_models.ipynb
│   ├── 10_statistical_testing.ipynb
│   ├── 11_cross_validation.ipynb
│   └── 12_blockchain_integration.ipynb
├── outputs/
│   ├── figures/              # All generated figures
│   ├── results/              # Metrics, predictions, McNemar results
│   └── cv_cache/             # Cross-validation fold cache
├── app.py                    # Flask demo server
├── demo.html                 # Frontend
├── .env                      # Infura API key (not committed)
└── requirements.txt
```

---

## Setup

```bash
git clone <repo>
cd cryptoguard
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
INFURA_KEY=your_infura_key_here
```

---

## Running the Demo

```bash
python app.py
```

Then open `http://localhost:5000` in your browser. All eight models load on startup — expect about 2–3 minutes before the feed becomes active, depending on your machine. The demo classifies messages from a rotating feed automatically and supports manual input via the text box at the bottom.

Note: run with `python app.py` directly, not `flask run` — the model loading thread only starts from the `__main__` block.

---

## Training

Training notebooks are designed to run on Google Colab (GPU required for transformers). Upload `data/` to `MyDrive/CryptoGuard/data/` before running.

Open `08.00-Train-Models.ipynb` on Colab and change the three config variables at the top of cell 1 for each run:

| Run | MODEL_TYPE | CORPUS | ADAPT_FROM |
|-----|-----------|--------|------------|
| 1 | tfidf | general | None |
| 2 | tfidf | blockchain | None |
| 3 | distilbert | general | None |
| 4 | bert | general | None |
| 5 | roberta | general | None |
| 6 | distilbert | blockchain | "distilbert_general" |
| 7 | bert | blockchain | "bert_general" |
| 8 | roberta | blockchain | "roberta_general" |

Restart the kernel between runs. Checkpoints save to `MyDrive/CryptoGuard/models/`.

---

## Evaluation

With all eight models downloaded locally, run the evaluation notebooks in order:

```
09_evaluate_models.ipynb      → 40 metrics, confusion matrices, ROC curves, F1 charts
10_statistical_testing.ipynb  → McNemar tests across all 28 model pairs
11_cross_validation.ipynb     → 5-fold CV on blockchain corpus (TF-IDF only locally)
12_blockchain_integration.ipynb → Live Sepolia testnet connection
```

---

## Key Results

| Model | General F1 | Blockchain F1 |
|-------|-----------|--------------|
| TF-IDF General | 0.9789 | 0.6684 |
| TF-IDF Blockchain | 0.6222 | 0.9754 |
| DistilBERT General | 0.9875 | 0.5832 |
| BERT General | 0.9907 | 0.6019 |
| RoBERTa General | 0.9906 | 0.6302 |
| DistilBERT Blockchain | 0.9535 | 0.9861 |
| BERT Blockchain | 0.9699 | 0.9899 |
| RoBERTa Blockchain | 0.9773 | **0.9962** |

General-trained models drop ~0.38 F1 on blockchain data across all three architectures. Domain adaptation recovers to F1 ≥ 0.986 in all cases. McNemar's tests confirm every general-to-adapted comparison is significant at p < 0.001.

---

## Environment

```
Python          3.13
PyTorch         2.12.0
Transformers    5.8.1
scikit-learn    1.8.0
pandas          3.0.3
NumPy           2.4.5
Matplotlib      3.10.9
Seaborn         0.13.2
Flask           3.x
web3            6.x
```

---

## Data

**General corpus** — Nazario phishing corpus + TREC07 spam corpus. 16,000 samples, balanced 50/50, 80/10/10 split.

**Blockchain corpus** — Synthetically generated using Claude, ChatGPT, Gemini, Grok, and DeepSeek. Raw corpus of 15,621 samples cleaned down to 7,952 balanced samples after deduplication and length filtering. 80/10/10 split.

Neither dataset contains personally identifiable information. Both are used for academic research purposes only.

---

## Supervisor

Dr. Rameez Asif, University of East Anglia