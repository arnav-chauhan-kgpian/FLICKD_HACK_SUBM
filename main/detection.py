import os
import torch
import json
from PIL import Image
from ultralytics import YOLO

# -------- CONFIG -------- #
MODEL_PATH = r"D:\flickd\models\best(4).pt"
FRAMES_ROOT = r"D:\flickd\frames"
CROPS_ROOT = r"D:\flickd\detection_results\crops"
DETECTIONS_ROOT = r"D:\flickd\detection_results\detections"
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.15  # vertical overlap threshold for co-ords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
model = YOLO(MODEL_PATH).to(device)
FASHION_CLASSES = model.names

# Create output directories
os.makedirs(CROPS_ROOT, exist_ok=True)
os.makedirs(DETECTIONS_ROOT, exist_ok=True)

YOLOS_TO_PRODUCT_TYPE = {
    'shirt, blouse': 'Shirt',
    'top, t-shirt, sweatshirt': 'Top',
    'sweater': 'Top',
    'cardigan': 'Top',
    'jacket': 'Jacket',
    'vest': 'Top',
    'pants': 'Trouser',
    'shorts': 'Shorts',
    'skirt': 'Skirt',
    'coat': 'Jacket',
    'dress': 'Dress',
    'jumpsuit': 'Jumpsuit',
    'cape': 'Top',
    'glasses': None, 'tie': None, 'belt': None, 'shoe': None, 'bag, wallet': None,
    'watch': None, 'hat': None, 'sleeve': None, 'pocket': None,
    'tights, stockings': None, 'sock': None, 'epaulette': None, 'neckline': None,
    'zipper': None, 'buckle': None, 'scarf': 'Scarf', 'hood': None, 'collar': None,
    'flower': None, 'ruffle': None, 'rivet': None, 'fringe': None,
    'bow': None, 'ribbon': None, 'sequin': None, 'bead': None, 'tassel': None,
    'umbrella': None, 'lapel': None, 'headband, head covering, hair accessory': None
}

def iou_y_overlap(box1, box2):
    _, y1a, _, y2a = box1
    _, y1b, _, y2b = box2
    y_overlap = max(0, min(y2a, y2b) - max(y1a, y1b))
    min_height = min(y2a - y1a, y2b - y1b)
    return y_overlap / (min_height + 1e-6)

def is_top(label):
    return label in {"Top"}

def is_bottom(label):
    return label in {"Trouser", "Shorts", "Skirt"}

def detect_and_crop_items(frame_path, crop_dir, conf_threshold=CONF_THRESHOLD):
    image = Image.open(frame_path).convert("RGB")
    results = model(image, conf=conf_threshold)[0]
    
    detected_items = []
    tops, bottoms = [], []

    for i, (box, score, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
        raw_label = FASHION_CLASSES[int(cls)]
        mapped_label = YOLOS_TO_PRODUCT_TYPE.get(raw_label.lower())

        if mapped_label is None:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image.crop((x1, y1, x2, y2))
        crop_filename = f"{os.path.basename(frame_path)[:-4]}_{mapped_label}_{i}.jpg"
        crop_path = os.path.join(crop_dir, crop_filename)
        crop.save(crop_path)

        det = {
            "label": raw_label,
            "mapped_label": mapped_label,
            "confidence": round(score.item(), 3),
            "bbox": [x1, y1, x2, y2],
            "class_id": int(cls),
            "crop_path": crop_path
        }
        detected_items.append(det)

        if is_top(mapped_label):
            tops.append(det)
        elif is_bottom(mapped_label):
            bottoms.append(det)

    # Co-ord detection
    co_ord_detections = []
    for top in tops:
        for bottom in bottoms:
            if iou_y_overlap(top["bbox"], bottom["bbox"]) > IOU_THRESHOLD:
                x1 = min(top["bbox"][0], bottom["bbox"][0])
                y1 = min(top["bbox"][1], bottom["bbox"][1])
                x2 = max(top["bbox"][2], bottom["bbox"][2])
                y2 = max(top["bbox"][3], bottom["bbox"][3])

                crop = image.crop((x1, y1, x2, y2))
                filename = f"{os.path.basename(frame_path)[:-4]}_Co-ord_{len(co_ord_detections)}.jpg"
                crop_path = os.path.join(crop_dir, filename)
                crop.save(crop_path)

                co_ord_detections.append({
                    "label": "Co-ord",
                    "mapped_label": "Co-ord",
                    "confidence": round(min(top["confidence"], bottom["confidence"]), 3),
                    "bbox": [x1, y1, x2, y2],
                    "class_id": -1,
                    "crop_path": crop_path,
                    "parts": [top["label"], bottom["label"]]
                })

    return detected_items + co_ord_detections

def process_video_frames(video_id, frames_dir):
    print(f"\n🎥 Processing: {video_id}")
    crop_dir = os.path.join(CROPS_ROOT, video_id)
    os.makedirs(crop_dir, exist_ok=True)

    detections_list = []

    for fname in sorted(os.listdir(frames_dir)):
        if not fname.lower().endswith(".jpg"):
            continue

        frame_path = os.path.join(frames_dir, fname)
        detections = detect_and_crop_items(frame_path, crop_dir)

        if detections:
            detections_list.append({
                "frame": fname,
                "detections": detections
            })

        print(f"✅ {fname} | {len(detections)} mapped items")

    output_json = os.path.join(DETECTIONS_ROOT, f"{video_id}.json")
    with open(output_json, "w") as f:
        json.dump(detections_list, f, indent=2)

    print(f"📄 Saved detection results to: {output_json}")

def main():
    video_folders = [d for d in os.listdir(FRAMES_ROOT) if os.path.isdir(os.path.join(FRAMES_ROOT, d))]

    if not video_folders:
        print("⚠️ No video folders found in frames!")
        return

    for video_id in sorted(video_folders):
        frames_dir = os.path.join(FRAMES_ROOT, video_id)
        process_video_frames(video_id, frames_dir)

    print("\n🎯 All detections complete.")

if __name__ == "__main__":
    main()
