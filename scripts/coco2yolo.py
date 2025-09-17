import json
import os

# --- CONFIG ---
json_file = "annotations.json"  # path to your COCO JSON
images_dir = "images"            # folder containing your images
labels_dir = "labels"            # folder to save YOLO .txt files

os.makedirs(labels_dir, exist_ok=True)

# --- LOAD JSON ---
with open(json_file) as f:
    data = json.load(f)

# --- BUILD IMAGE ID MAP ---
image_info = {img['id']: img for img in data['images']}

# --- PROCESS ANNOTATIONS ---
for ann in data['annotations']:
    img_id = ann['image_id']
    img = image_info[img_id]
    img_width, img_height = img['width'], img['height']

    # COCO bbox: [x_top_left, y_top_left, width, height]
    x, y, w, h = ann['bbox']

    # Convert to YOLO format
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    class_id = ann['category_id']  # make sure these match your YOLO class indices

    # Create label file path
    file_name = os.path.splitext(img['file_name'])[0] + ".txt"
    label_path = os.path.join(labels_dir, file_name)

    # Append annotation line
    with open(label_path, 'a') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("Conversion complete! YOLO .txt files saved in:", labels_dir)
