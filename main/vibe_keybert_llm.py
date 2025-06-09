import os, json, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

# ---------- CONFIG ---------- #
CATALOG_PATH = r"D:\flickd\data_preprocessing\catalog_cleaned.csv"
OUTPUT_DIR = r"D:\flickd\outputs"
TRANSCRIPT_DIR = r"D:\flickd\data\video_transcripts"

VIBES = ["coquette", "clean girl", "cottagecore", "streetcore", "y2k", "boho", "party glam"]

VIBE_KEYWORDS = {
    "coquette": ["romantic", "feminine", "bow", "ruffle", "delicate", "puff sleeve", "lace", "smocked", "tie-up", "pink", "pastel"],
    "clean girl": ["minimal", "neutral", "beige", "white", "fresh", "structured", "polished", "simple", "sleek"],
    "cottagecore": ["floral", "butterfly", "garden", "organic", "meadow", "cotton", "natural", "sage", "countryside"],
    "streetcore": ["denim", "oversized", "urban", "cargo", "hoodie", "relaxed", "boyfriend", "graphic", "wide leg"],
    "y2k": ["metallic", "cyber", "neon", "geometric", "color blocked", "tech", "futuristic", "bold"],
    "boho": ["bohemian", "free", "flowy", "tiered", "wrap", "strappy", "earthy", "vacation", "fringe", "maxi"],
    "party glam": ["sequin", "glitter", "bodycon", "fitted", "night", "cocktail", "cutout", "backless", "plunging", "satin", "gold"]
}

COLOR_VIBE_MAP = {
    "coquette": ["pink", "lavender", "powder blue", "lilac", "pastel yellow"],
    "boho": ["olive", "brown", "khaki", "sage green", "olive green", "rust", "chocolate brown", "tan", "mustard"],
    "party glam": ["red", "maroon", "wine red", "coral", "purple"],
    "y2k": ["orange", "purple", "turquoise blue", "cobalt blue", "turquoise"],
    "cottagecore": ["green", "light green", "sage green", "peach"],
    "clean girl": ["beige", "off white", "cream", "ivory", "ecru white"],
    "streetcore": ["black", "grey", "navy", "denim", "dark blue", "charcoal"]  
}

PRINT_VIBE_MAP = {
    "cottagecore": ["floral", "checks", "embroidered", "tropical", "floral and fruit", "floral and solid"],
    "clean girl": ["solid", "self design", "soild"],
    "boho": ["ethnic motifs", "embroidered", "abstract", "embellished", "embroidery"],
    "streetcore": ["stripe", "striped", "stripes", "graphic", "camouflage", "typography"],
    "y2k": ["animal", "geometric", "colourblocked", "lurex stripe"],
    "coquette": ["polka dots", "polka", "schiffli", "embellished"],
    "party glam": ["embellished", "lurex stripe", "abstract"]
}

# ---------- MODELS ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer("all-mpnet-base-v2", device=device)
vibe_embeds = {vibe: encoder.encode(keywords, convert_to_tensor=True) for vibe, keywords in VIBE_KEYWORDS.items()}

# ---------- HELPERS ---------- #
def safe_eval(x):
    try:
        return json.loads(x.replace("'", '"')) if isinstance(x, str) else x
    except:
        return {}

def tokenize(text, min_len=3):
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    return [t for t in tokens if len(t) >= min_len]

def exact_match_score(tokens):
    scores = {vibe: 0.0 for vibe in VIBES}
    token_set = set(tokens)
    for vibe, keywords in VIBE_KEYWORDS.items():
        scores[vibe] = sum(1 for kw in keywords if kw in token_set)
    return scores

def semantic_similarity_score(tokens, top_k=5):
    if not tokens: return {vibe: 0.0 for vibe in VIBES}
    token_embeds = encoder.encode(tokens, convert_to_tensor=True)
    scores = {}
    for vibe, vibe_embed in vibe_embeds.items():
        sim = util.cos_sim(token_embeds, vibe_embed)
        max_sims = torch.topk(sim, k=min(top_k, sim.shape[1]), dim=1).values
        scores[vibe] = float(torch.mean(max_sims))
    return scores

def color_print_score(tags):
    tags = safe_eval(tags)
    color = tags.get("Colour", "").lower()
    print_ = tags.get("Print", "").lower()
    scores = {vibe: 0.0 for vibe in VIBES}
    for vibe in VIBES:
        scores[vibe] += sum(0.6 for c in COLOR_VIBE_MAP[vibe] if c in color)
        scores[vibe] += sum(0.4 for p in PRINT_VIBE_MAP[vibe] if p in print_)
    return scores

def transcript_score(video_id):
    path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.txt")
    if not os.path.exists(path): return {vibe: 0.0 for vibe in VIBES}
    with open(path, encoding="utf-8") as f:
        content = f.read()
    tokens = tokenize(content)
    return semantic_similarity_score(tokens)

def normalize(scores):
    max_score = max(scores.values()) if scores else 1
    return {k: v / max_score if max_score > 0 else 0.0 for k, v in scores.items()}

def ensemble(rule, sem, cp, transcript, weights=(0.6, 0.25, 0.05, 0.1)):
    return {
        vibe: weights[0] * rule.get(vibe, 0.0) +
              weights[1] * sem.get(vibe, 0.0) +
              weights[2] * cp.get(vibe, 0.0) +
              weights[3] * transcript.get(vibe, 0.0)
        for vibe in VIBES
    }

def get_top_vibes(scores, threshold=0.8, max_vibes=3):
    sorted_v = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [v for v, s in sorted_v if s >= threshold][:max_vibes] or [sorted_v[0][0]]

# ---------- MAIN ---------- #
def run_vibe_classification():
    df = pd.read_csv(CATALOG_PATH)
    df["product_id"] = df["product_id"].astype(str)

    for file in tqdm(os.listdir(OUTPUT_DIR), desc="🧠 Classifying vibes"):
        if not file.endswith(".json"): continue
        video_id = file.replace(".json", "")
        path = os.path.join(OUTPUT_DIR, file)
        with open(path) as f: data = json.load(f)

        matched_ids = [str(p["matched_product_id"]) for p in data.get("products", [])]
        products = df[df["product_id"].isin(matched_ids)]

        if products.empty:
            data["vibes"] = ["clean girl"]
            with open(path, "w") as f: json.dump(data, f, indent=2)
            continue

        all_tokens = []
        color_scores = []
        for _, row in products.iterrows():
            txt = f"{row.get('title', '')} {row.get('description', '')} {row.get('product_collections', '')}"
            all_tokens += tokenize(txt)
            color_scores.append(color_print_score(row.get("product_tags", "{}")))

        rule = normalize(exact_match_score(all_tokens))
        sem = normalize(semantic_similarity_score(all_tokens))
        cp = normalize({v: np.mean([s.get(v, 0.0) for s in color_scores]) for v in VIBES})
        ts = normalize(transcript_score(video_id))

        final_scores = ensemble(rule, sem, cp, ts)
        top_vibes = get_top_vibes(final_scores)

        data["vibes"] = top_vibes
        data["confidence_scores"] = {k: round(v, 3) for k, v in final_scores.items()}

        with open(path, "w") as f: json.dump(data, f, indent=2)
        print(f"✅ {video_id}: {top_vibes}")

    print("🎯 Vibe classification completed.")

if __name__ == "__main__":
    run_vibe_classification()
