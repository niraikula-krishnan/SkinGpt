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

Upload a skin photo → get a disease prediction → chat with an AI assistant for deeper context.

**Try it live:** [huggingface.co/spaces/niraikula-krishnan/SkinGpt](https://huggingface.co/spaces/niraikula-krishnan/SkinGpt)  
**GitHub:** [github.com/niraikula-krishnan/SkinGpt](https://github.com/niraikula-krishnan/SkinGpt)

---

## What can it do?

- Detect skin conditions from an image (22 diseases supported)
- Show confidence score and top 3 predictions
- Chat with an AI assistant (LLaMA via Groq) backed by a disease knowledge base (RAG)
- **Session users** — starts with **User 1**; add more users with **+ Add** when needed
- **Per-user analytics** — track diseases discussed per user; view a bar chart on request
- **Chat sidebar** — multiple chats per browser tab (cleared on page refresh)

> **Disclaimer:** For learning and information only. Always consult a clinician for real diagnosis.

---

## How to use

1. Open the chat page and start as **User 1** (created automatically)
2. Upload a clear, close-up photo of the skin area (optional) and type a question
3. Read the prediction and AI explanation; use suggestion chips for follow-ups
4. Click **+ Add** in the sidebar to create additional session users (e.g. Patient A, Dr. Smith)
5. Ask **"Provide the analytics of this session"** to see a per-user disease chart
6. **Refresh the page** to reset chats, users, and analytics for a new session

**Supported image types:** JPG, PNG, GIF, BMP

---

## Run on your computer

**1. Install packages**

```powershell
cd C:\Users\nirai\ML_pjt\code
pip install -r requirements.txt
```

**2. Create a `.env` file** in the `code` folder:

```env
GROQ_API_KEY=your-groq-key-here
FLASK_SECRET_KEY=any-random-string

# Optional — MySQL for per-user analytics (falls back to memory if unset)
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_DATABASE=skingpt
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

**3. (Optional) Set up MySQL**

```powershell
python setup_mysql.py
```

**4. Start the app**

```powershell
python app.py
```

**5. Open in browser:** [http://127.0.0.1:7860](http://127.0.0.1:7860)

---

## Main files

| File | What it does |
|------|--------------|
| `app.py` | Flask routes, LLM chat, analytics |
| `db.py` | Per-user analytics (MySQL or in-memory fallback) |
| `rag.py` | TF-IDF retrieval over disease notes |
| `skin_model.h5` | Trained EfficientNet model (~17 MB) |
| `class_names.json` | Disease labels the model knows |
| `schema.sql` | MySQL table definition |
| `setup_mysql.py` | Create database and tables locally |
| `templates/` | Landing page and chat UI |
| `rag/` | Disease `.txt` files for chat context |
| `train.py` | Retrain the model |
| `predict.py` | Predict from terminal (no browser) |
| `deploy_hf.ps1` | Upload to Hugging Face Space via CLI |

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

## Push to GitHub & Hugging Face

From the `code` folder:

```powershell
cd C:\Users\nirai\ML_pjt\code

git add .
git commit -m "Your commit message"
git push origin main
git push space main
```

- **GitHub** remote: `origin` → `github.com/niraikula-krishnan/SkinGpt`
- **Hugging Face** remote: `space` → `huggingface.co/spaces/niraikula-krishnan/SkinGpt`

Use a GitHub PAT or Hugging Face **Write** token when prompted.

---

## Deploy on Hugging Face (free)

Live demo: [niraikula-krishnan-skingpt.hf.space](https://niraikula-krishnan-skingpt.hf.space)

1. Push to the Space (`git push space main` or `.\deploy_hf.ps1`)
2. In **Settings → Secrets**, add:
   - `GROQ_API_KEY` — your Groq API key
   - `FLASK_SECRET_KEY` — any random string
3. Wait a few minutes for the Docker Space to rebuild

MySQL is not required on Hugging Face — analytics use in-memory storage per container.

---

## Built with

Python · Flask · TensorFlow · Groq (LLaMA 3.3) · LangChain · MySQL (optional) · Chart.js · HTML/CSS/JS
