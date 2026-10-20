from pathlib import Path
from ultralytics import YOLO
import os
import numpy as np

# Load YOLO model
ROOT = Path(__file__).resolve().parents[1]
model = YOLO(str(ROOT / "models" / "best.pt"))

# Folders
image_folder = str(ROOT / "data")
output_folder = str(ROOT / "results")
os.makedirs(output_folder, exist_ok=True)

# Store total unique fruit count
total_fruit_count = 0

# Dictionary to store fruit positions for uniqueness check
fruit_positions = {}

# Process images in pairs (front & back views)
pair_ids = sorted({p.stem.removeprefix("front_img") for p in Path(image_folder).glob("front_img*.jpg")}, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
for i in pair_ids:
    front_img = f"front_img{i}.jpg"
    back_img = f"back_img{i}.jpg"
    
    front_img_path = os.path.join(image_folder, front_img)
    back_img_path = os.path.join(image_folder, back_img)
    
    if os.path.exists(front_img_path) and os.path.exists(back_img_path):
        # Process front image
        front_results = model(front_img_path)
        for result in front_results:
            output_path = os.path.join(output_folder, f"{os.path.splitext(front_img)[0]}_result.jpg")
            result.save(output_path)
        
        # Process back image
        back_results = model(back_img_path)
        for result in back_results:
            output_path = os.path.join(output_folder, f"{os.path.splitext(back_img)[0]}_result.jpg")
            result.save(output_path)
        
        print(f"✅ Processed: {front_img} and {back_img}")
    else:
        print(f"❌ Missing image pair: {front_img} or {back_img}")

print("\n🎉 All image pairs processed successfully! Results saved in 'results/' folder.")