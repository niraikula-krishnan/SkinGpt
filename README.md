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

# SkinGPT

Upload a skin photo → get a disease prediction → chat with an AI assistant for more info.

**Try it live:** [huggingface.co/spaces/niraikula-krishnan/SkinGpt](https://huggingface.co/spaces/niraikula-krishnan/SkinGpt)  
**GitHub:** [github.com/niraikula-krishnan/SkinGpt](https://github.com/niraikula-krishnan/SkinGpt)

---

## What can it do?

- Detect skin conditions from an image (22 diseases supported)
- Show confidence score and top 3 predictions
- Answer follow-up questions using a medical knowledge base + AI chat (LLaMA via Groq)

> **Disclaimer:** For learning and information only. Always see a doctor for real diagnosis.

---

## Run on your computer

**1. Install packages**
```powershell
cd C:\Users\nirai\ML_pjt\code
pip install -r requirements.txt
```

**2. Set your Groq API key** (needed for chat)
```powershell
$env:GROQ_API_KEY = "your-key-here"
```
Get a free key at [console.groq.com](https://console.groq.com)

**3. Start the app**
```powershell
python app.py
```

**4. Open in browser:** [http://127.0.0.1:7860](http://127.0.0.1:7860)

Click **Start Predicting**, upload a skin image, and type a message.

---

## How to use

1. Upload a clear, close-up photo of the skin area
2. Type a question (e.g. "What is this condition?")
3. The app predicts the disease and the AI explains it
4. Keep chatting — it remembers your conversation

**Supported image types:** JPG, PNG, GIF, BMP

---

## Main files

| File | What it does |
|------|--------------|
| `app.py` | Main web app |
| `skin_model.h5` | Trained AI model (~17 MB) |
| `class_names.json` | Disease names the model knows |
| `rag/` | Disease info used for chat answers |
| `templates/` | Web pages |
| `train.py` | Retrain the model |
| `predict.py` | Predict from terminal (no browser) |

---

## Train your own model (optional)

Put images in folders — one folder per disease:
```
dataset/
  Acne/
    photo1.jpg
  Eczema/
    photo2.jpg
```

Then run:
```powershell
python train.py --dataset "path\to\your\dataset.zip" --epochs 5
```

Or use the [Colab notebook](Colab_SkinDisease_Training.ipynb) to train in the cloud.

---

## Deploy on Hugging Face (free)

Live demo: [niraikula-krishnan-skingpt.hf.space](https://niraikula-krishnan-skingpt.hf.space)

1. Push this repo to [Hugging Face Spaces](https://huggingface.co/spaces) (Docker SDK)
2. In **Settings → Secrets**, add:
   - `GROQ_API_KEY` — your Groq API key
   - `FLASK_SECRET_KEY` — any random text string
3. The Space rebuilds automatically — wait a few minutes

---

## Built with

Python · Flask · TensorFlow · Groq (LLaMA 3.3) · LangChain · HTML/CSS/JS
