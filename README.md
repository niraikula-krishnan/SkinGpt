---
title: SkinGPT
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SkinGPT AI — Skin Disease Classifier & Chat Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Colab_SkinDisease_Training.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/niraikula-krishnan/SkinGpt)
[![GitHub Repo](https://img.shields.io/badge/GitHub-SkinGpt-181717?logo=github)](https://github.com/niraikula-krishnan/SkinGpt)

An AI-powered web app that detects skin diseases from uploaded images and lets you chat with an LLM assistant for deeper insights.

---

## What it does

- Predicts skin conditions from a photo using a trained EfficientNet model
- Shows confidence score, session accuracy, and top-3 predictions
- Chat with an AI assistant (LLaMA 3.3 via Groq) that remembers the full conversation
- RAG (Retrieval-Augmented Generation) — answers questions from a verified knowledge base of 22 skin diseases
- Follow-up question chips and typing indicator for a smooth chat experience
- Medical disclaimer and trust badges for credibility

---

## Project Files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend — handles routes, ML prediction, LangChain LLM |
| `rag.py` | RAG engine — loads disease text files, retrieves relevant chunks using TF-IDF |
| `templates/index.html` | Landing page |
| `templates/chat.html` | Chat page |
| `train.py` | Model training script |
| `predict.py` | Terminal-based prediction tool |
| `skin_model.h5` | Trained EfficientNet model weights |
| `class_names.json` | Maps model output index to disease name |
| `rag/` | Folder containing 22 disease knowledge `.txt` files |
| `requirements.txt` | All required Python packages |

---

## Diseases Supported (22)

Acne, Actinic Keratosis, Benign Tumors, Bullous, Candidiasis, Drug Eruption, Eczema, Infestations & Bites, Lichen, Lupus, Moles, Psoriasis, Rosacea, Seborrheic Keratoses, Skin Cancer, Sun Damage, Tinea, Unknown/Normal, Vascular Tumors, Vasculitis, Vitiligo, Warts

---

## Setup

1. Go to the project folder:
```powershell
cd C:\Users\nirai\ML_pjt\code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure these files are present:
- `skin_model.h5`
- `class_names.json`

---

## Run the App

```powershell
python app.py
```

Open in browser: `http://127.0.0.1:5000`

- The landing page gives an overview of the app
- Click **Start Predicting** to go to the chat page
- Attach a skin image and type a message to get a prediction and AI response

---

## How the Chat Works

1. Upload a skin image — the model predicts the disease
2. RAG retrieves relevant information from the disease knowledge base
3. LangChain passes the full conversation history + RAG context to LLaMA 3.3
4. The AI responds with accurate, context-aware answers
5. Follow-up question chips appear after each reply for quick interaction

---

## Train the Model

Organise your dataset like this:
```
dataset/
  Acne/
    img1.jpg
  Eczema/
    img2.jpg
```

Then run:
```powershell
python train.py --dataset "C:\Users\nirai\ML_pjt\SkinDisease.zip" --epochs 5
```

If you pass a zip file it will be extracted automatically.

---

## Terminal Prediction (No Browser)

```powershell
python predict.py
```

Type the image path when prompted. Type `quit` to exit.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML Model | TensorFlow, EfficientNet B3 |
| LLM | LLaMA 3.3 70B via Groq API |
| LLM Framework | LangChain (`ChatGroq`, `RunnableWithMessageHistory`) |
| RAG | Scikit-learn TF-IDF cosine similarity |
| Frontend | HTML, CSS, Vanilla JavaScript |

---

## 🚀 Deploy to Hugging Face Spaces (Free)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on%20Spaces-available-brightgreen)](https://huggingface.co/spaces/niraikula-krishnan/skingpt)

### One-click deployment:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. Name: `SkinGpt` | SDK: **Docker**
3. Connect your GitHub repo: `https://github.com/niraikula-krishnan/SkinGpt`
4. Set these **Secrets** in Space settings:
   - `GROQ_API_KEY` — your Groq API key (get one free at console.groq.com)
   - `FLASK_SECRET_KEY` — any random string
5. The Space will auto-build and give you a public URL like:  
   `https://niraikula-krishnan-skingpt.hf.space`  
   (or open [huggingface.co/spaces/niraikula-krishnan/SkinGpt](https://huggingface.co/spaces/niraikula-krishnan/SkinGpt))

> **Note:** The model file (`skin_model.h5`, ~17MB) is already in the repo.  
> HF Spaces auto-detects `requirements.txt` and runs `app.py`.

---

## 📓 Google Colab Notebook

Train the model yourself in the cloud:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Colab_SkinDisease_Training.ipynb)

1. Open the notebook via the badge above
2. Upload your `disease.zip` dataset
3. Run all cells — it trains a CNN and saves `skin_model.h5`
4. Download the trained model and use it with this app

---

## Notes

- Use clear, well-lit, close-up photos of the affected skin area for best results
- The app is for **informational purposes only** — always consult a dermatologist
- Supported image formats: JPG, JPEG, PNG, GIF, BMP
- Conversation memory is stored per browser tab session using LangChain's `ChatMessageHistory`
