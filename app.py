import json
import os
import uuid
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image as PILImage
from rag import retrieve
import db

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

preprocess_input = tf.keras.applications.efficientnet.preprocess_input
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
CONFIDENCE_THRESHOLD = 0.40

# Optional OpenAI integration: set OPENAI_API_KEY in environment to enable
# Optional Groq API key: set GROQ_API_KEY environment variable to enable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

client = None

def get_llm_client():
    global client
    if client is not None:
        return client
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        client = ChatGroq(api_key=key, model="llama-3.3-70b-versatile")
        return client
    except Exception:
        return None

get_llm_client()

app = Flask(__name__)
db.init_db()
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, UPLOAD_FOLDER)
app.secret_key = os.environ.get(
    'FLASK_SECRET_KEY',
    'skin-disease-chatbot-secret'
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_class_names() -> list[str] | None:
    path = os.path.join(app.root_path, 'class_names.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_model() -> tf.keras.Model:
    path = os.path.join(app.root_path, 'skin_model.h5')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    return tf.keras.models.load_model(path)


model = load_model()
class_names = load_class_names()
# Running accuracy tracker: list of per-prediction confidence scores
_prediction_confidences: list[float] = []

def get_session_accuracy() -> str | None:
    """Average confidence across all predictions this session."""
    if not _prediction_confidences:
        return None
    return f'{sum(_prediction_confidences) / len(_prediction_confidences):.2f}'


TEMPERATURE = 0.5  # <1 sharpens predictions, increases confidence separation

def predict_image(image_path: str) -> dict:
    base_img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    # TTA: original + flips + crops at 5 positions
    w, h = base_img.size
    crop_size = int(min(w, h) * 0.9)
    offsets = [(0,0),(w-crop_size,0),(0,h-crop_size),(w-crop_size,h-crop_size),((w-crop_size)//2,(h-crop_size)//2)]
    crops = [base_img.crop((x, y, x+crop_size, y+crop_size)).resize((224,224)) for x, y in offsets]

    tta_imgs = [
        base_img,
        base_img.transpose(PILImage.FLIP_LEFT_RIGHT),
        base_img.transpose(PILImage.FLIP_TOP_BOTTOM),
    ] + crops

    logits_list = []
    for a in tta_imgs:
        arr = tf.keras.utils.img_to_array(a)
        arr = np.expand_dims(arr, 0)
        arr = preprocess_input(arr)
        raw = model.predict(arr, verbose=0)[0]  # raw softmax outputs
        # convert back to logit-space, apply temperature, re-softmax
        log_p = np.log(np.clip(raw, 1e-9, 1.0)) / TEMPERATURE
        log_p -= log_p.max()
        sharpened = np.exp(log_p) / np.exp(log_p).sum()
        logits_list.append(sharpened)

    preds = np.mean(logits_list, axis=0)
    valid_class_names = class_names if class_names and len(class_names) == preds.shape[-1] else None

    top_idxs = preds.argsort()[-3:][::-1]
    best_idx = int(top_idxs[0])
    best_confidence = float(preds[best_idx] * 100)
    low_confidence = best_confidence < CONFIDENCE_THRESHOLD * 100

    _prediction_confidences.append(best_confidence)

    return {
        'label': valid_class_names[best_idx] if valid_class_names else f'Class {best_idx}',
        'confidence': f'{best_confidence:.2f}',
        'accuracy': get_session_accuracy(),
        'low_confidence': low_confidence,
        'note': 'Low confidence — try a clearer, well-lit image.' if low_confidence else None,
        'top3': [
            {
                'label': valid_class_names[int(idx)] if valid_class_names else f'Class {int(idx)}',
                'score': f'{float(preds[int(idx)] * 100):.2f}',
            }
            for idx in top_idxs
        ],
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# In-memory session store: session_id -> list of {"role", "content"} dicts
_session_store: dict[str, list[dict]] = {}
_session_meta: dict[str, dict] = {}
MAX_HISTORY = 20  # keep last 20 messages (10 turns)


def _display_disease(name: str) -> str:
    return name.replace('_', ' ')


def _find_diseases_ordered(text: str) -> list[str]:
    """Return disease display names in the order they appear in the message."""
    if not class_names or not text:
        return []
    lower = text.lower()
    matches: list[tuple[int, str]] = []
    for name in class_names:
        display = _display_disease(name)
        variants = (
            name.lower(),
            display.lower(),
            name.lower().replace('_', ' '),
            name.lower().replace('_', ''),
        )
        positions = [lower.find(v) for v in variants if v in lower]
        if positions:
            matches.append((min(positions), display))
    matches.sort(key=lambda item: item[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for _, display in matches:
        if display not in seen:
            seen.add(display)
            ordered.append(display)
    return ordered


def _record_disease(user_number: str, disease: str) -> None:
    db.record_disease(user_number, disease, _display_disease)


def _update_user_analytics(user_number: str, user_message: str, prediction_label: str = '') -> None:
    if prediction_label:
        _record_disease(user_number, prediction_label)
    if user_message and not _is_analytics_request(user_message):
        for disease in _find_diseases_ordered(user_message):
            _record_disease(user_number, disease)


def _is_analytics_request(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return False
    triggers = (
        'provide the analytics',
        'provide analytics',
        'show analytics',
        'show the analytics',
        'session analytics',
        'analytics of this session',
        'analysis of this session',
        'session statistics',
        'session stats',
        'show session chart',
    )
    return any(p in t for p in triggers)


def _session_context_block(session_id: str, user_number: str) -> str:
    """Brief summary so the LLM can answer follow-ups in the same chat session."""
    parts: list[str] = []
    meta = _session_meta.get(session_id, {})
    last_pred = meta.get('last_prediction')
    if last_pred:
        parts.append(
            f"Latest image prediction in this chat: {last_pred['label']} "
            f"({last_pred['confidence']}% confidence)"
        )
    counts = db.get_analytics(user_number)
    if counts:
        summary = ', '.join(f"{name} ({count}x)" for name, count in counts.items())
        parts.append(f"Diseases discussed for {user_number}: {summary}")
    if not parts:
        return ''
    return (
        'Session context (use for follow-up questions like "it", "this", or "that condition"):\n'
        + '\n'.join(parts)
    )


def _update_session_meta(session_id: str, prediction: dict | None, user_message: str) -> None:
    meta = _session_meta.setdefault(session_id, {})
    if prediction and prediction.get('label'):
        meta['last_prediction'] = {
            'label': _display_disease(prediction['label']),
            'confidence': prediction.get('confidence', ''),
        }
    if user_message:
        diseases = _find_diseases_ordered(user_message)
        if diseases:
            meta['last_topic'] = diseases[-1]


@app.route('/api/users', methods=['GET', 'POST'])
def api_users():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        label = (data.get('user_label') or request.form.get('user_label', '')).strip()
        if not label:
            return jsonify({'error': 'user_label is required'}), 400
        user = db.create_user(label)
        return jsonify({'user': user, 'users': db.list_users()}), 201
    return jsonify({'users': db.list_users(), 'mysql': db.db_active()})


@app.route('/api/session/reset', methods=['POST'])
def api_session_reset():
    db.reset_all_users()
    return jsonify({'ok': True, 'message': 'Session reset — User 1 restored'})


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('chat.html')

    user_message = request.form.get('message', '').strip()
    session_id = request.form.get('session_id', 'default')
    user_number = request.form.get('user_number', '').strip()
    if user_number and not db.user_exists(user_number):
        user_number = ''

    if not user_number:
        return jsonify({
            'reply': 'Please add and select a user in the sidebar before sending messages.',
            'prediction': None,
            'image_url': None,
            'accuracy': None,
            'show_chart': False,
            'analytics': {},
            'user_number': '',
        })
    image = request.files.get('image')
    prediction = None
    image_url = None

    if image and image.filename != '' and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        unique_name = f'{uuid.uuid4().hex}_{filename}'
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        image.save(save_path)
        try:
            prediction = predict_image(save_path)
            image_url = url_for('uploaded_file', filename=unique_name)
        except Exception as e:
            prediction = {'error': str(e)}

    system_prompt = (
        'You are a helpful medical assistant that discusses skin disease predictions. '
        'You are in an ongoing chat session — read the conversation history and session context '
        'and answer follow-up questions accordingly (e.g. if the user asks "what causes it?", '
        'refer to the disease or prediction discussed earlier in this session). '
        'When verified disease knowledge is provided below, use it as your PRIMARY source to answer accurately. '
        'When an image prediction is provided, include it in your reasoning but do not give definitive medical advice. '
        'Always encourage users to consult a clinician for diagnosis and treatment.'
    )

    context = ''
    prediction_label = ''
    if prediction and prediction.get('label'):
        prediction_label = prediction['label']
        context += f"Image prediction: {prediction_label} ({prediction['confidence']}%); Top3: "
        context += ', '.join([f"{i['label']} ({i['score']}%)" for i in prediction.get('top3', [])])
    elif prediction and prediction.get('error'):
        context += f"Image prediction failed: {prediction['error']}"

    # RAG retrieval
    rag_context = retrieve(user_message, prediction_label)
    if rag_context:
        context = rag_context + ('\n\n' + context if context else '')

    assistant_reply = ""

    # Get or create session history
    if session_id not in _session_store:
        _session_store[session_id] = []

    history = _session_store[session_id]

    if _is_analytics_request(user_message):
        counts = db.get_analytics(user_number)
        label = db.get_user_label(user_number)
        if counts:
            assistant_reply = (
                f'Here is analytics for {label}. '
                'Each bar shows how many separate times you discussed each disease '
                '(back-to-back questions about the same disease count once until you switch to another).'
            )
        else:
            assistant_reply = (
                f'No analytics yet for {label}. '
                'Upload a skin image or ask about a disease first, then request analytics again.'
            )
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})
        if len(history) > MAX_HISTORY * 2:
            history[:2] = []
        return jsonify({
            'reply': assistant_reply,
            'prediction': prediction,
            'image_url': image_url,
            'accuracy': prediction.get('accuracy') if prediction else get_session_accuracy(),
            'show_chart': bool(counts),
            'analytics': counts,
            'user_number': user_number,
        })

    _update_user_analytics(user_number, user_message, prediction_label)
    _update_session_meta(session_id, prediction, user_message)

    session_ctx = _session_context_block(session_id, user_number)
    full_system_prompt = system_prompt
    if session_ctx:
        full_system_prompt += f'\n\n{session_ctx}'

    # Build current user message with context appended for the LLM
    current_user_content = user_message
    if context:
        current_user_content += f'\n\n{context}'

    # Build messages: system + stored history + current user message
    messages = [
        {"role": "system", "content": full_system_prompt},
        *history[-MAX_HISTORY:],
        {"role": "user", "content": current_user_content}
    ]

    llm = get_llm_client()
    if llm is not None:
        try:
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
            response = llm.invoke(lc_messages)
            assistant_reply = response.content.strip()
        except Exception:
            try:
                fallback = ChatGroq(api_key=os.environ.get("GROQ_API_KEY", "").strip(), model="llama-3.1-8b-instant")
                response = fallback.invoke(lc_messages)
                assistant_reply = response.content.strip()
            except Exception as e2:
                assistant_reply = f"(LLM error: {str(e2)})"
    else:
        assistant_reply = "LLM is not configured. Please set a valid GROQ_API_KEY in the environment."

    # Store exchange in session history (include prediction note so follow-ups stay in context)
    stored_user = user_message
    if prediction and prediction.get('label'):
        note = (
            f"[Uploaded skin image — prediction: {_display_disease(prediction['label'])} "
            f"({prediction['confidence']}% confidence)]"
        )
        stored_user = f"{user_message}\n{note}".strip() if user_message else note

    history.append({"role": "user", "content": stored_user})
    history.append({"role": "assistant", "content": assistant_reply})

    # Trim old history if needed
    if len(history) > MAX_HISTORY * 2:
        history[:2] = []

    return jsonify({
        'reply': assistant_reply,
        'prediction': prediction,
        'image_url': image_url,
        'accuracy': prediction.get('accuracy') if prediction else get_session_accuracy(),
        'analytics': db.get_analytics(user_number),
        'user_number': user_number,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)