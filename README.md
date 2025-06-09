# 🎯 Flickd AI Hackathon Submission

**Smart Tagging & Vibe Classification Engine (Backend MVP)**

## 📌 Project Goal

To build a **fully functional backend MVP** for **Flickd’s Smart Tagging & Vibe Classification Engine**, which:

* Detects and classifies fashion products frame-by-frame in videos.
* Extracts aesthetic "vibes" for the entire video using multimodal input (visual, text, transcript).
* Produces structured JSON outputs for downstream UX/UX design, recommendation, and analytics use.

---

## 🗂️ Repository Structure

```bash
FLICKD_HACK_SUBM
├── data/
│   ├── video_transcripts/  #.txt transcripts per video
│   ├── videos/  #.mp4 videos
│   ├── videos-20250604T053958Z-1-001/  #original videos dataset given
│   ├── images.csv  #product ids + image urls 
|   ├── product_data.xlsx  #product ids + product descriptions
│   └── vibeslist.json  #vibes considered for classification
├── data_preprocessing/          
│   ├── catalog_images.csv/  #downloaded images
│   ├── catalog_cleaned.csv  #dataset after preprocessing
│   ├── catalog_with_paths.csv  #dataset after image_downloader.py
│   ├── catalog.csv  #dataset after catalog_creator.py
|   ├── catalog_creator.py
|   ├── image_downloader.py 
│   └── simple_preprocessing.py
├── detection_results/                   
|   ├── crops/  #detected product crops
│   └── detections/
├── frames/ #extracted frames
├── main/                 
│   ├── detection.py    # carry out detections
|   ├── frame_extraction.py    # extract frames
|   ├── matcher.py    # match products
│   └── vibe_classifier.py #classify vibes per video
├── models/                  
|   ├── clip-vit-base-patch32/    
|   ├── best.pt    
│   └── yolov8n.pt
├── outputs/  #Final JSONs with vibe classifications per video
├── yolov8_finetuning/ #finetuned yolov8 model for fashion detection
│   ├── best(1).pt
|   ├── best(2).pt
│   ├── best(3).pt
│   ├── best(4).pt #same as best.pt present in models/
│   ├── yolov8n.pt    
|   ├── yolov8_1.ipynb  
|   ├── yolov8_2.ipynb 
│   └── yolov8_3.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 🔧 1. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FLICKD_HACK_SUBM.git
cd FLICKD_HACK_SUBM

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Required Inputs

Place the following in the `data/` directory:

* `product_data.xlsx`
* `images.csv`
* `video_transcripts/`
* `videos`
* `vibeslist.json`

### 3. Extract Frames(if not already)

``` bash
python main/frame_extraction.py
```
### 4. Run Product Detection (YOLOv8)

```bash
python main/detection.py
```

* Output saved to `/crops/<video_id>/`
* Each frame gets cropped product images

### 5. Run Product Matching

```bash
python main/matcher.py
```

* Matches detected items with cleaned catalog using embedding-based matching
  
### ⚙️ 6. Classify Vibes (Per Video)

```bash
python main/vibe_classifier.py
```

* Final vibe classifcations saved to: `/outputs/` as JSONs

---

## 📊 Sample Output Format (`outputs/<video_id>.json`)

```json
{
  "video_id": "video123",
  "vibes": ["cottagecore", "boho"],
  "confidence_scores": {
    "coquette": 0.12,
    "clean girl": 0.03,
    "cottagecore": 0.92,
    "boho": 0.88,
    ...
  },
  "products": [
    {
      "matched_product_id": "123456",
      "mapped_label": "Dress",
      "confidence": 0.89,
      "crop_path": "crops/video123/Dress_0.jpg"
    },
    ...
  ]
}
```

---

## 📽️ Demo Video

🔗 [Loom demo (≤ 5 mins)](https://loom.com/share/your-demo-link)

---

## 📝 Evaluation JSONs

All final outputs are stored under:

```
outputs/
├── video123.json
├── video124.json
└── ...
```

---

## Vibe Classification Logic 

Priority is:

1. **Exact keyword matches** (title, description, collection)
2. **Semantic similarity** using Sentence Transformers
3. **Color & print features** from product\_tags
4. **Transcript context** (if available)

> All scores are ensemble-weighted, normalized, and final vibes are chosen based on top confidences.

---

## 📌 Future Enhancements

* LLM-based caption refinement (Mistral-7B / Mixtral)
* REST API endpoints via FastAPI
* Automatic video → frames extractor
* Streamlit or React-based dashboard

---

## 🤝 Credits

This project was built as a part of the **Flickd AI Hackathon**.
By: \[Arnav Chauhan (3rd Year UG at IIT Kharagpur)]
GitHub: [@arnav-chauhan-kgpian](https://github.com/arnav-chauhan-kgpian)

---
