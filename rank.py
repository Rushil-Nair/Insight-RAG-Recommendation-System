import math
import re
import time
from datetime import datetime
import numpy as np
import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize shared resources globally to avoid reloading
try:
    tokenization_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    # Fallback if spacy model missing
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    tokenization_model = spacy.load("en_core_web_sm", disable=["parser", "ner"])

stemmer = PorterStemmer()
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True, stop_words='english', ngram_range=(1, 2), max_features=10000
)

# Constants
MATCH_TEXT_2_INT_PATTERN = r"^(\d+(\.\d+)?)(k|kk|k\+)?$"
MATCH_WITH_2_BOUNDS_PATTERN = r'(?:between\s+)?\$?(\d[\d,\.]*\s*(?:k|kk)?)\s*(?:-+|~|to|and)\s*\$?(\d[\d,\.]*\s*(?:k|kk)?)'
MATCH_WITH_UPPER_BOUND_PATTERN = r'(?:under|below|less than|at most|≤|<=)\s*\$?(\d[\d,\.]*\s*(?:k|kk)?)'
MATCH_APPROX_PATTERN = r'(?:around|about|approx(?:imately)?|near|nearly)\s*\$?(\d[\d,\.]*\s*(?:k|kk)?)'
MATCH_EXP_PATTERN_1 = r'\$?(\d[\d,\.]*\s*(?:k|kk)?)\s*(?:budget|budgets?)'
MATCH_EXP_PATTERN_2 = r'(?:budget|budgets?)\s*\$?(\d[\d,\.]*\s*(?:k|kk)?)'
THOUSAND = 1000.0
SECOND_OF_DAY = 86400.0
MAX_STAR_RATING = 5.0

ASPECT_LEXICON = {
    "size": ["mini", "compact", "small", "under‐counter", "narrow", "slim", "dorm", "space saving"],
    "capacity": ["capacity", "storage", "liters", "cu ft", "cubic feet", "cans", "bottles", "holds", "fits"],
    "noise": ["quiet", "low noise", "silent", "hum", "buzz", "whisper", "noise level", "noisy", "sound"],
    "energy": ["energy", "efficient", "efficiency", "save power", "low watt", "energy star", "power consumption"],
    "design": ["glass door", "stainless", "black", "white", "handle", "reversible door", "sleek", "modern design"],
    "cooling": ["cool", "cold", "cooling", "freeze", "freezer", "chill", "temperature", "ice", "frost"],
    "price": ["price", "budget", "cost", "expensive", "cheap", "worth", "value", "cost-effective"],
    "material": ["material", "build quality", "plastic", "metal", "durable", "premium", "finish", "texture"]
}

def min_max_normalization(values, lower=1, upper=99, epsilon=1e-9):
    arr = np.array(values, dtype=float)
    low, high = np.percentile(arr, [lower, upper])
    clipped = np.clip(arr, low, high)
    if np.isclose(high, low):
        return np.ones_like(arr) * 0.5
    return (clipped - low) / (high - low + epsilon)

def parse_budget(query):
    lower_query = query.lower()
    def parse_money(text):
        text = text.replace(",", "").strip()
        regex_match = re.match(MATCH_TEXT_2_INT_PATTERN, text)
        if not regex_match: return None
        val = float(regex_match.group(1))
        if regex_match.group(3): val *= THOUSAND
        return val

    regex_match = re.search(MATCH_WITH_2_BOUNDS_PATTERN, lower_query)
    if regex_match:
        bound_a = parse_money(regex_match.group(1))
        bound_b = parse_money(regex_match.group(2))
        if bound_a is not None and bound_b is not None:
            return (bound_a + bound_b) / 2.0

    regex_match = re.search(MATCH_WITH_UPPER_BOUND_PATTERN, lower_query)
    if regex_match:
        money = parse_money(regex_match.group(1))
        if money is not None: return money

    regex_match = re.search(MATCH_APPROX_PATTERN, lower_query)
    if regex_match:
        money = parse_money(regex_match.group(1))
        if money is not None: return money

    regex_match = re.search(MATCH_EXP_PATTERN_1, lower_query)
    if regex_match:
        money = parse_money(regex_match.group(1))
        if money is not None: return money

    regex_match = re.search(MATCH_EXP_PATTERN_2, lower_query)
    if regex_match:
        money = parse_money(regex_match.group(1))
        if money is not None: return money
    return None

def stem(word):
    exceptions = {'thing': 'thing', 'things': 'thing', 'address': 'address', 'business': 'business', 'glass': 'glass', 'stainless': 'stainless'}
    lw = word.lower()
    if lw in exceptions: return exceptions[lw]
    stemmed = stemmer.stem(lw)
    if len(stemmed) < 3: return lw
    return stemmed

def tokenize(text):
    if not text: return []
    doc = tokenization_model(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space or token.is_stop: continue
        tl = token.lemma_.lower()
        if re.match(r"[a-z0-9\+\-]+", tl): tokens.append(tl)
    return tokens

def aspect_match_score(query, text):
    lower_query = (query or "").lower()
    lower_text = (text or "").lower()
    query_tokens = tokenize(lower_query)
    text_tokens = tokenize(lower_text)
    query_stemming_set = set(stem(word) for word in query_tokens)
    text_stemming_set = set(stem(word) for word in text_tokens)
    found = 0
    total = 0
    for aspect, key_words_set in ASPECT_LEXICON.items():
        aspect_stems = set(stem(word) for word in key_words_set)
        if query_stemming_set & aspect_stems:
            total += 1
            if text_stemming_set & aspect_stems:
                found += 1
    if total == 0: return 1.0
    return found / total

def price_fit_score(price, anchor, price_std=None, sigma_ratio=0.3, sigma_std_factor=0.5):
    if price is None or anchor is None or price <= 0 or anchor <= 0: return 0.5
    sigma_base = sigma_ratio * anchor
    if price_std is not None and price_std > 0:
        sigma_base = max(sigma_base, sigma_std_factor * float(price_std))
    over_scale = 0.8
    under_scale = 1.2
    sigma = (over_scale if price > anchor else under_scale) * sigma_base
    return math.exp(-((price - anchor) ** 2) / (2 * sigma * sigma))

def bayesian_rating(avg_rate, rate_count, global_avg_rate=4.2, m=50.0):
    if avg_rate is None: avg_rate = global_avg_rate
    if rate_count is None: rate_count = 0
    return (global_avg_rate * m + avg_rate * rate_count) / (m + rate_count)

def parse_timestamp(x):
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    try: return datetime.fromisoformat(x.replace('Z', '')).timestamp()
    except Exception: return None

def recency_score(last_ts, now_ts=None, lambda_per_day=1 / 180):
    if last_ts is None: return 0.5
    if now_ts is None: now_ts = time.time()
    age_days = max(0.0, (now_ts - last_ts) / SECOND_OF_DAY)
    return math.exp(-lambda_per_day * age_days)

def fit_tfidf_corpus(texts):
    tfidf_vectorizer.fit(texts)

def tfidf_text_vector(text):
    if not text: return np.zeros(tfidf_vectorizer.max_features, dtype=float)
    X = tfidf_vectorizer.transform([text])
    vec = X.toarray()[0]
    norm = np.linalg.norm(vec)
    if norm < 1e-9: return vec
    return vec / norm

def rank(product_ids, similarities, products, reviews, query, top_k_candidates=200, final_k=10, weights=None, learned_weights=None, lambda_per_day_by_category=None):
    pairs = sorted(zip(product_ids, similarities), key=lambda x: x[1], reverse=True)[:top_k_candidates]
    if not pairs: return []

    budget = parse_budget(query)
    candidate_prices = [products[pid].get('price') for pid, _ in pairs if products.get(pid, {}).get('price') is not None]
    price_anchor = budget if budget else (float(np.median(candidate_prices)) if candidate_prices else None)
    price_std = (float(np.std(candidate_prices)) if candidate_prices else None)

    ratings = []
    popularities = []
    recencies = []
    price_fits = []
    aspect_overlaps = []

    for pid, sim in pairs:
        meta_data = products.get(pid, {})
        star_rating = meta_data.get('rating') or meta_data.get('average_rating')
        review_count = meta_data.get('review_count') or meta_data.get('rating_number')
        last_timestamp = parse_timestamp(meta_data.get('last_updated_time') or meta_data.get('last_review_time'))
        price = meta_data.get('price')
        title = meta_data.get('title', '')
        text_for_aspect = (title or '') + ' ' + (meta_data.get('summary', '') or '')

        bayesian_rating_score = bayesian_rating(star_rating, review_count) / MAX_STAR_RATING
        popularity = math.log1p(review_count) if isinstance(review_count, (int, float)) and review_count >= 0 else 0.0
        
        category = meta_data.get('category')
        if lambda_per_day_by_category and category in lambda_per_day_by_category:
            recency = recency_score(last_timestamp, lambda_per_day=lambda_per_day_by_category[category])
        else:
            recency = recency_score(last_timestamp)

        price_fit = price_fit_score(price, price_anchor, price_std=price_std)
        aspect_overlap = aspect_match_score(query, text_for_aspect)

        ratings.append(bayesian_rating_score)
        popularities.append(popularity)
        recencies.append(recency)
        price_fits.append(price_fit)
        aspect_overlaps.append(aspect_overlap)

    pop_norm = min_max_normalization(popularities)

    default_weights = {
        "alpha": 0.50, "beta": 0.20, "gamma": 0.10, 
        "delta": 0.05, "epsilon": 0.10, "zeta": 0.05, "mmr_lambda": 0.3
    }
    if weights is None: weights = default_weights.copy()
    else:
        merged = default_weights.copy()
        merged.update(weights)
        weights = merged
    if learned_weights: weights.update(learned_weights)

    final_rating_scores = []
    for i, (pid, sim) in enumerate(pairs):
        score = (
            weights["alpha"] * float(sim) +
            weights["beta"] * float(ratings[i]) +
            weights["gamma"] * float(pop_norm[i]) +
            weights["delta"] * float(recencies[i]) +
            weights["epsilon"] * float(price_fits[i]) +
            weights["zeta"] * float(aspect_overlaps[i])
        )
        final_rating_scores.append((pid, score))

    def mmr_cosine_similarity(vec_a, vec_b, epsilon=1e-9):
        if vec_a is None or vec_b is None: return 0.0
        arr_a = np.asarray(vec_a, dtype=float)
        arr_b = np.asarray(vec_b, dtype=float)
        norm_a = np.linalg.norm(arr_a)
        norm_b = np.linalg.norm(arr_b)
        if norm_a < epsilon or norm_b < epsilon: return 0.0
        return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))

    id_2_vec = {}
    for pid, _ in pairs:
        if pid in reviews and reviews.get(pid) is not None:
            review_vector = reviews.get(pid)
            arr = np.asarray(review_vector, dtype=float)
            vec_norm = np.linalg.norm(arr)
            epsilon = 1e-9
            id_2_vec[pid] = arr / (vec_norm + epsilon)
        else:
            meta_data = products.get(pid, {})
            text = (meta_data.get('title') or '')
            id_2_vec[pid] = tfidf_text_vector(text)

    id_2_score = dict(final_rating_scores)
    mmr_lambda = float(weights.get("mmr_lambda", 0.8))

    selected = []
    candidates = [pid for pid, _ in sorted(final_rating_scores, key=lambda x: x[1], reverse=True)]

    while len(selected) < min(final_k, len(candidates)):
        best_pid = None
        best_mmr_score = -1e9
        for pid in candidates:
            relevance = id_2_score[pid]
            if not selected:
                mmr_val = relevance
            else:
                similarities_to_selected = [mmr_cosine_similarity(id_2_vec.get(pid), id_2_vec.get(s)) for s in selected]
                max_sim = max(similarities_to_selected) if similarities_to_selected else 0.0
                mmr_val = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            
            if mmr_val > best_mmr_score:
                best_pid = pid
                best_mmr_score = mmr_val
        
        if best_pid is None: break
        selected.append(best_pid)
        candidates.remove(best_pid)

    return [(pid, id_2_score[pid]) for pid in selected]