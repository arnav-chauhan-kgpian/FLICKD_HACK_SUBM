import os
import subprocess
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = r'D:\flickd\models\yolov8n.pt'
FFMPEG_PATH = r'C:\PATH_programs\ffmpeg.exe'
VIDEOS_DIR = r"D:\flickd\data\videos"
OUTPUT_ROOT = r"D:\flickd\frames"
CONF_THRESHOLD = 0.1

def validate_paths():
    if not os.path.exists(FFMPEG_PATH):
        raise FileNotFoundError(f"FFmpeg not found at: {FFMPEG_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at: {MODEL_PATH}")
    if not os.path.exists(VIDEOS_DIR):
        raise FileNotFoundError(f"Videos folder not found at: {VIDEOS_DIR}")

def extract_keyframes_ffmpeg(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        FFMPEG_PATH, '-i', video_path,
        '-skip_frame', 'nokey',
        '-vsync', '0',
        '-q:v', '2',
        '-frame_pts', '1',
        os.path.join(output_dir, 'frame_%05d.jpg')
    ]
    subprocess.run(command, check=True)

def filter_person_frames(input_dir, conf_threshold=CONF_THRESHOLD):
    model = YOLO(MODEL_PATH)
    selected_frames = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        frame_path = os.path.join(input_dir, fname)
        results = model.predict(frame_path, conf=conf_threshold, classes=[0], imgsz=640, verbose=False)
        if len(results[0].boxes) > 0:
            selected_frames.append(frame_path)
    return selected_frames

def process_video(video_file):
    video_id = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(OUTPUT_ROOT, video_id)
    print(f"\n🎥 Processing {video_id}...")
    
    extract_keyframes_ffmpeg(video_file, output_dir)
    person_frames = filter_person_frames(output_dir)

    print(f"✅ {len(person_frames)} useful frames retained from {video_id}")
    return video_id, person_frames

def main():
    validate_paths()
    all_videos = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(".mp4")]
    
    if not all_videos:
        print("⚠️ No .mp4 files found in the videos directory!")
        return

    overall_stats = {}

    for video in all_videos:
        video_path = os.path.join(VIDEOS_DIR, video)
        try:
            video_id, retained = process_video(video_path)
            overall_stats[video_id] = {
                "total_frames": len(os.listdir(os.path.join(OUTPUT_ROOT, video_id))),
                "person_frames": len(retained)
            }
        except Exception as e:
            print(f"❌ Failed to process {video}: {e}")

    print("\n📊 Summary:")
    for vid, stats in overall_stats.items():
        print(f"  {vid}: {stats['person_frames']}/{stats['total_frames']} person-frames retained")


if __name__ == "__main__":
    main()
