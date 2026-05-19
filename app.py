#CryptoGuard served by Flask

import os
import re
import sys
import time
import pickle
import threading
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = BASE_DIR / 'models'

app = Flask(__name__)
CORS(app)

#global model holders
MODELS = {}          # {model_key: loaded model dict}
models_ready = False
load_status  = {"status": "loading", "message": "Initialising..."}

MODEL_REGISTRY = [
    ("tfidf_general",         "tfidf"),
    ("tfidf_blockchain",      "tfidf"),
    ("distilbert_general",    "transformer"),
    ("bert_general",          "transformer"),
    ("roberta_general",       "transformer"),
    ("distilbert_blockchain", "transformer"),
    ("bert_blockchain",       "transformer"),
    ("roberta_blockchain",    "transformer"),
]

DISPLAY_NAMES = {
    "tfidf_general":         "TF-IDF General",
    "tfidf_blockchain":      "TF-IDF Blockchain",
    "distilbert_general":    "DistilBERT General",
    "bert_general":          "BERT General",
    "roberta_general":       "RoBERTa General",
    "distilbert_blockchain": "DistilBERT Blockchain",
    "bert_blockchain":       "BERT Blockchain",
    "roberta_blockchain":    "RoBERTa Blockchain",
}

#text cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

#model loading
def load_models():
    global models_ready, load_status

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = torch.device('cpu')

        for model_key, model_type in MODEL_REGISTRY:
            checkpoint = MODELS_DIR / model_key
            load_status = {
                "status":  "loading",
                "message": f"Loading {DISPLAY_NAMES[model_key]}..."
            }
            print(f"  Loading {model_key}...", end=" ")

            try:
                if model_type == "tfidf":
                    with open(checkpoint / "vectorizer.pkl", "rb") as f:
                        vec = pickle.load(f)
                    with open(checkpoint / "classifier.pkl", "rb") as f:
                        clf = pickle.load(f)
                    MODELS[model_key] = {
                        "type":       "tfidf",
                        "vectorizer": vec,
                        "clf":        clf,
                    }
                else:
                    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
                    model     = AutoModelForSequenceClassification.from_pretrained(
                        str(checkpoint)
                    )
                    model.to(device)
                    model.eval()
                    MODELS[model_key] = {
                        "type":      "transformer",
                        "tokenizer": tokenizer,
                        "model":     model,
                        "device":    device,
                    }
                print("✓")
            except Exception as e:
                print(f"✗  {e}")
                MODELS[model_key] = None

        loaded = sum(1 for v in MODELS.values() if v is not None)
        models_ready = True
        load_status  = {
            "status":  "ready",
            "message": f"All {loaded}/8 models loaded and ready."
        }
        print(f"\n✓ {loaded}/8 models loaded.")

    except Exception as e:
        load_status = {"status": "error", "message": str(e)}
        print(f"✗ Model loading failed: {e}")


#inference helpers
def run_tfidf(model_dict, text):
    cleaned = clean_text(text)
    vec     = model_dict['vectorizer'].transform([cleaned])
    prob    = float(model_dict['clf'].predict_proba(vec)[0][1])
    return {
        "probability":    round(prob, 4),
        "prediction":     int(prob >= 0.5),
        "label":          "Phishing" if prob >= 0.5 else "Legitimate",
        "confidence_pct": round(prob * 100, 1),
    }


def run_transformer(model_dict, text, return_attention=False):
    import torch
    cleaned   = clean_text(text)
    tokenizer = model_dict['tokenizer']
    model     = model_dict['model']
    device    = model_dict['device']

    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='pt',
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=return_attention,
        )
        probs         = torch.softmax(outputs.logits, dim=1)[0]
        phishing_prob = float(probs[1])

    result = {
        "probability":    round(phishing_prob, 4),
        "prediction":     int(phishing_prob >= 0.5),
        "label":          "Phishing" if phishing_prob >= 0.5 else "Legitimate",
        "confidence_pct": round(phishing_prob * 100, 1),
    }

    if return_attention and outputs.attentions:
        #average attention across all heads, last layer, [CLS] token
        last_layer  = outputs.attentions[-1]          # (1, heads, seq, seq)
        avg_heads   = last_layer[0].mean(dim=0)       # (seq, seq)
        cls_attn    = avg_heads[0]                    # (seq,) attention from [CLS]
        cls_attn    = cls_attn.cpu().numpy().tolist()

        #get tokens
        token_ids = inputs['input_ids'][0].cpu().numpy().tolist()
        tokens    = tokenizer.convert_ids_to_tokens(token_ids)

        #normalise attention weights to [0,1]
        attn_arr  = np.array(cls_attn)
        if attn_arr.max() > 0:
            attn_norm = (attn_arr / attn_arr.max()).tolist()
        else:
            attn_norm = attn_arr.tolist()

        result['attention'] = {
            "tokens":  tokens,
            "weights": attn_norm,
        }

    return result


def run_model(model_key, text, return_attention=False):
    model_dict = MODELS.get(model_key)
    if model_dict is None:
        return {"error": f"{model_key} not loaded"}
    if model_dict['type'] == 'tfidf':
        return run_tfidf(model_dict, text)
    else:
        return run_transformer(model_dict, text, return_attention=return_attention)


#routes
@app.route('/')
def index():
    return send_from_directory(str(BASE_DIR), 'demo.html')


@app.route('/status')
def status():
    return jsonify(load_status)


@app.route('/classify', methods=['POST'])
def classify():
    if not models_ready:
        return jsonify({
            "error":  "Models still loading",
            "status": load_status["message"]
        }), 503

    data = request.get_json()
    text = data.get('text', '').strip()
    if not text or len(text) < 5:
        return jsonify({"error": "Text too short"}), 400

    t0      = time.time()
    results = {}

    for model_key, _ in MODEL_REGISTRY:
        results[model_key] = run_model(model_key, text)

    elapsed = round((time.time() - t0) * 1000, 1)

    #ensemble: majority vote across all 8 models
    votes = sum(
        1 for k in results
        if isinstance(results[k], dict) and results[k].get('prediction') == 1
    )
    ensemble_label = "Phishing" if votes > 4 else "Legitimate"

    return jsonify({
        "text":           text[:200],
        "results":        results,
        "ensemble_votes": votes,
        "ensemble_label": ensemble_label,
        "inference_ms":   elapsed,
    })


#server startup
if __name__ == '__main__':
    print("Starting CryptoGuard demo server (8-model mode)...")
    print(f"Models directory: {MODELS_DIR}")
    thread = threading.Thread(target=load_models, daemon=True)
    thread.start()
    print("Server starting at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)