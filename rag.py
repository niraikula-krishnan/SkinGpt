import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAG_DIR = os.path.join(os.path.dirname(__file__), 'rag')
CHUNK_SIZE = 10  # lines per chunk

# disease name aliases covering all RAG txt files 
DISEASE_ALIASES = {
    'acne': ['acne', 'acne vulgaris', 'pimple', 'pimples', 'blackhead', 'whitehead', 'breakout'],
    'actinic_keratosis': ['actinic keratosis', 'actinic keratoses', 'solar keratosis', 'solar keratoses', 'sun keratosis'],
    'benign_tumors': ['benign tumor', 'benign tumours', 'benign growth', 'skin tag', 'lipoma', 'sebaceous cyst', 'dermatofibroma', 'benign lesion'],
    'bullous': ['bullous', 'blister', 'blisters', 'pemphigoid', 'pemphigus', 'blistering'],
    'candidiasis': ['candidiasis', 'candida', 'yeast infection', 'thrush', 'fungal infection'],
    'drugeruption': ['drug eruption', 'drug reaction', 'drug rash', 'medication rash', 'adverse drug', 'toxicoderma', 'fixed drug'],
    'eczema': ['eczema', 'dermatitis', 'atopic dermatitis', 'contact dermatitis', 'skin inflammation', 'dry skin rash', 'eczematous'],
    'infestations_bites': ['infestation', 'bite', 'bites', 'scabies', 'lice', 'bed bug', 'flea', 'mite', 'insect bite', 'parasitic'],
    'lichen': ['lichen', 'lichen planus', 'lichen sclerosus', 'lichen simplex', 'lichenification'],
    'lupus': ['lupus', 'sle', 'systemic lupus', 'cutaneous lupus', 'butterfly rash', 'malar rash', 'discoid lupus'],
    'moles': ['mole', 'moles', 'nevus', 'nevi', 'melanocytic', 'atypical mole', 'dysplastic', 'birthmark'],
    'psoriasis': ['psoriasis', 'psoriatic', 'scaly skin', 'plaque psoriasis', 'scalp psoriasis', 'guttate', 'nail psoriasis'],
    'rosacea': ['rosacea', 'rosacea redness', 'facial redness', 'acne rosacea', 'ocular rosacea', 'couperose', 'red face'],
    'seborrh_keratoses': ['seborrheic keratosis', 'seborrheic keratoses', 'sebaceous hyperplasia', 'barnacle', 'senile wart', 'brown growth'],
    'skincancer': ['skin cancer', 'skin tumor', 'malignant', 'basal cell', 'squamous cell', 'melanoma', 'bcc', 'scc', 'carcinoma', 'malignancy'],
    'sun_sunlight_damage': ['sun damage', 'sunlight damage', 'photoaging', 'solar damage', 'uv damage', 'sunburn', 'actinic damage', 'photodamage'],
    'tinea': ['tinea', 'ringworm', 'athlete foot', 'jock itch', 'dermatophyte', 'fungal skin', 'tinea corporis', 'tinea pedis', 'tinea cruris', 'tinea capitis'],
    'unknown_normal': ['normal skin', 'healthy skin', 'no condition', 'unknown', 'benign mole', 'normal mole', 'no issue', 'clear skin'],
    'vascular_tumors': ['vascular tumor', 'vascular lesion', 'hemangioma', 'angioma', 'port wine', 'cherry angioma', 'pyogenic granuloma', 'blood vessel'],
    'vasculitis': ['vasculitis', 'vessel inflammation', 'purpura', 'petechiae', 'livedo', 'leukocytoclastic', 'blood vessel inflammation'],
    'vitiligo': ['vitiligo', 'white patch', 'depigmentation', 'loss of pigment', 'leukoderma', 'white spots', 'pigment loss'],
    'warts': ['wart', 'warts', 'verruca', 'verrucae', 'hpv', 'viral wart', 'plantar wart', 'genital wart', 'skin tag viral'],
}


def _load_chunks() -> tuple[list[str], list[str]]:
    """Load all RAG txt files and split into chunks. Returns (chunks, sources)."""
    chunks, sources = [], []
    if not os.path.isdir(RAG_DIR):
        return chunks, sources
    for fname in os.listdir(RAG_DIR):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(RAG_DIR, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        for i in range(0, len(lines), CHUNK_SIZE):
            chunk = ' '.join(lines[i:i + CHUNK_SIZE])
            if chunk:
                chunks.append(chunk)
                sources.append(fname.replace('.txt', '').capitalize())
    return chunks, sources


# load once at import time
_chunks, _sources = _load_chunks()
_vectorizer = None
_tfidf_matrix = None

if _chunks:
    _vectorizer = TfidfVectorizer(stop_words='english')
    _tfidf_matrix = _vectorizer.fit_transform(_chunks)


def _detect_disease(query: str) -> str | None:
    """Return disease name if query clearly mentions one of the RAG diseases."""
    q = query.lower()
    for disease, aliases in DISEASE_ALIASES.items():
        if any(alias in q for alias in aliases):
            return disease
    return None


def retrieve(query: str, prediction_label: str = '', top_k: int = 4) -> str:
    """
    Retrieve the most relevant RAG chunks for a query.
    If a prediction label is provided, bias retrieval towards that disease.
    Returns a formatted context string to inject into the LLM prompt.
    """
    if not _chunks or _vectorizer is None:
        return ''

    # combine query with prediction label for better matching
    full_query = f"{query} {prediction_label}".strip()

    # check if query or prediction maps to a known disease file
    detected = _detect_disease(full_query)
    if not detected and prediction_label:
        detected = _detect_disease(prediction_label)

    # if detected disease, filter chunks to that disease only
    if detected:
        indices = [i for i, s in enumerate(_sources) if s.lower() == detected.lower()]
        if indices:
            sub_chunks = [_chunks[i] for i in indices]
            sub_sources = [_sources[i] for i in indices]
            sub_vec = TfidfVectorizer(stop_words='english').fit_transform(sub_chunks)
            q_vec = TfidfVectorizer(stop_words='english').fit(sub_chunks).transform([full_query])
            sims = cosine_similarity(q_vec, sub_vec)[0]
            top_indices = sims.argsort()[-top_k:][::-1]
            results = [(sub_chunks[i], sub_sources[i], sims[i]) for i in top_indices if sims[i] > 0.01]
            if results:
                return _format_context(results)

    # fallback: search across all chunks
    q_vec = _vectorizer.transform([full_query])
    sims = cosine_similarity(q_vec, _tfidf_matrix)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    results = [(_chunks[i], _sources[i], sims[i]) for i in top_indices if sims[i] > 0.01]
    return _format_context(results)


def _format_context(results: list[tuple[str, str, float]]) -> str:
    if not results:
        return ''
    lines = ['--- Verified Disease Knowledge ---']
    for chunk, source, score in results:
        lines.append(f'[{source}]: {chunk}')
    lines.append('--- End of Knowledge ---')
    return '\n'.join(lines)