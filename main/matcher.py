import os
import json
import faiss
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# -------- Paths -------- #
CLIP_MODEL_PATH = r"D:\flickd\models\clip-vit-base-patch32"
CATALOG_PATH = r"D:\flickd\data_preprocessing\catalog_cleaned.csv"
CROPS_DIR = r"D:\flickd\detection_results\crops"
DETECTIONS_DIR = r"D:\flickd\detection_results\detections"
OUTPUT_DIR = r"D:\flickd\outputs"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load model -------- #
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device)
model.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- Helpers -------- #
def encode_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to open image: {image_path}, error: {e}")
        raise
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True)

def safe_parse_tags(tags):
    if isinstance(tags, dict):
        return tags
    try:
        return json.loads(tags.replace("'", '"')) if isinstance(tags, str) else {}
    except Exception:
        return {}

def build_catalog_index(catalog_df):
    embeddings = []
    product_ids = []
    for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df), desc="📦 Building catalog index"):
        path = row["local_path"]
        if not os.path.exists(path):
            print(f"⚠️ File not found, skipping: {path}")
            continue
        try:
            emb = encode_image(path)
            embeddings.append(emb.cpu().numpy())
            product_ids.append(row["product_id"])
        except Exception as e:
            print(f"❌ Error encoding {path}: {e}")
            continue
    if not embeddings:
        raise ValueError("No valid catalog embeddings generated.")
    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"✅ Added {len(product_ids)} products to index.")
    return index, product_ids

def match_crop(crop_path, index, product_ids):
    try:
        emb = encode_image(crop_path).cpu().numpy().astype("float32")
        D, I = index.search(emb, 1)
        similarity = float(D[0][0])
        match_id = product_ids[I[0][0]]

        if similarity > 0.9:
            return match_id, "Exact Match", similarity
        elif similarity >= 0.75:
            return match_id, "Similar Match", similarity
        else:
            return match_id, "No Match", similarity
    except Exception as e:
        print(f"❌ Error matching crop {crop_path}: {e}")
        return None

def process_video(video_id, index, product_ids, catalog_df):
    detection_path = os.path.join(DETECTIONS_DIR, f"{video_id}.json")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")

    if not os.path.exists(detection_path):
        print(f"⚠️ Detection file missing for video {video_id}, skipping.")
        return

    with open(detection_path, "r") as f:
        detections = json.load(f)

    matched_products = {}

    for frame in detections:
        for det in frame.get("detections", []):
            crop_path = det.get("crop_path")
            if not crop_path or not os.path.exists(crop_path):
                continue
            result = match_crop(crop_path, index, product_ids)
            if result:
                pid, match_type, sim = result

                if match_type == "No Match" or sim < 0.75:
                    continue  # Skip low confidence

                if pid in matched_products:
                    continue  # Avoid duplicates

                row = catalog_df[catalog_df["product_id"] == pid]
                color = "unknown"
                if not row.empty:
                    tags = safe_parse_tags(row.iloc[0].get("product_tags", {}))
                    color = tags.get("Colour", "unknown").lower()

                matched_products[pid] = {
                    "type": det.get("mapped_label", "unknown").lower(),
                    "color": color,
                    "match_type": match_type,
                    "matched_product_id": pid,
                    "confidence": round(sim, 3)
                }

    result = {
        "video_id": video_id,
        "products": list(matched_products.values())
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved: {output_path}")

# -------- Main -------- #
if __name__ == "__main__":
    print("📥 Loading catalog CSV...")
    catalog_df = pd.read_csv(CATALOG_PATH)
    print(f"📊 Catalog size before filtering: {len(catalog_df)}")

    catalog_df = catalog_df[
        catalog_df["local_path"].apply(lambda x: isinstance(x, str) and os.path.exists(x))
    ]
    print(f"✅ Catalog size after filtering: {len(catalog_df)}")
    print("🖼️ Sample paths:\n", catalog_df["local_path"].head())

    index, product_ids = build_catalog_index(catalog_df)

    video_ids = [f[:-5] for f in os.listdir(DETECTIONS_DIR) if f.endswith(".json")]
    print(f"🎬 Found {len(video_ids)} videos.")

    for video_id in sorted(video_ids):
        process_video(video_id, index, product_ids, catalog_df)
