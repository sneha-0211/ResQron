import argparse
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

# --- Utility Functions ---

def ensure_dir(path: Path):
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)

def write_label(file_path: Path, labels):
    """Write a list of YOLO labels to a .txt file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")

def get_image_size(image_path: Path):
    """Get the width and height of an image."""
    with Image.open(image_path) as im:
        return im.size

def move_files(files, split_name, out_dir):
    """Move image and label files to the appropriate train/val subdirectory."""
    for f_name in files:
        src_img = out_dir / "images" / f_name
        src_lbl = out_dir / "labels" / (Path(f_name).stem + ".txt")
        dst_img_dir = out_dir / "images" / split_name
        dst_lbl_dir = out_dir / "labels" / split_name
        ensure_dir(dst_img_dir)
        ensure_dir(dst_lbl_dir)
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img_dir / f_name))
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl_dir / (Path(f_name).stem + ".txt")))

# --- Format-Specific Converters ---

def convert_coco_format(json_path, out_dir, image_dir):
    """Handles COCO JSON part of VisDrone."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    class_names = sorted(list(set(categories.values())))
    class_to_id = {name: i for i, name in enumerate(class_names)}
    ann_by_img = {}
    for ann in data["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    all_items = []
    sources = {}
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"

    for img_id, anns in ann_by_img.items():
        img_info = images[img_id]
        file_name = img_info["file_name"]
        src_path = image_dir / file_name
        if not src_path.exists(): continue
        shutil.copy(src_path, out_img_dir / file_name)
        im_w, im_h = img_info.get("width"), img_info.get("height")
        if not im_w or not im_h: im_w, im_h = get_image_size(src_path)
        labels = []
        for ann in anns:
            cls_id = class_to_id[categories[ann["category_id"]]]
            x, y, w, h = ann["bbox"]
            xc, yc = x + w / 2, y + h / 2
            labels.append([cls_id, xc / im_w, yc / im_h, w / im_w, h / im_h])
        write_label(out_lbl_dir / (Path(file_name).stem + ".txt"), labels)
        all_items.append(file_name)
        sources[file_name] = {'dataset': 'visdrone-coco', 'source_ann': str(json_path)}
    return class_to_id, all_items, sources

def convert_txt_format(ann_dir, out_dir, image_dir):
    """Handles native VisDrone TXT format."""
    class_set, all_items, sources, temp_labels = set(), [], {}, {}
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"

    txt_files = sorted(list(ann_dir.rglob("*.txt")))
    for txt_file in txt_files:
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            cand_path = image_dir / (txt_file.stem + ext)
            if cand_path.exists(): image_path = cand_path; break
        if not image_path: continue
        shutil.copy(image_path, out_img_dir / image_path.name)
        w, h = get_image_size(image_path)
        labels = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6: continue
                xmin, ymin, w_box, h_box = map(float, parts[0:4])
                clsname = parts[5].strip()
                xc, yc = xmin + w_box / 2.0, ymin + h_box / 2.0
                labels.append((clsname, xc / w, yc / h, w_box / w, h_box / h))
                class_set.add(clsname)
        temp_labels[image_path.name] = labels
        all_items.append(image_path.name)
        sources[image_path.name] = {'dataset': 'visdrone-txt', 'source_ann': str(txt_file)}

    class_to_id = {name: i for i, name in enumerate(sorted(list(class_set)))}
    for filename, labels in temp_labels.items():
        yolo_labels = []
        for cls, xc, yc, w_box, h_box in labels:
            yolo_labels.append([class_to_id[cls], xc, yc, w_box, h_box])
        write_label(out_lbl_dir / (Path(filename).stem + ".txt"), yolo_labels)
    return class_to_id, all_items, sources

# --- Main Conversion Logic ---

def convert(args):
    """Main function to perform VisDrone to YOLO conversion."""
    ann_dir = Path(args.ann)
    image_dir = Path(args.images)
    out_dir = Path(args.out)
    ensure_dir(out_dir / "images")
    ensure_dir(out_dir / "labels")

    json_files = list(ann_dir.rglob("*.json"))
    if json_files:
        print(f"Info: Found VisDrone COCO JSON file: {json_files[0]}. Using it for conversion.")
        class_to_id, all_items, sources = convert_coco_format(json_files[0], out_dir, image_dir)
    else:
        txt_files = list(ann_dir.rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No VisDrone annotations (.json or .txt) found recursively in {ann_dir}")
        print(f"Info: Found {len(txt_files)} VisDrone .txt annotation files. Processing...")
        class_to_id, all_items, sources = convert_txt_format(ann_dir, out_dir, image_dir)
    
    if not all_items:
        raise RuntimeError("No VisDrone images were processed. Check paths and file extensions.")

    # --- Split dataset and write config files ---
    if args.test_size == 0.0:
        print("Info: test_size is 0. Assigning all items to the training set.")
        train_items = all_items
        val_items = []
    else:
        train_items, val_items = train_test_split(all_items, test_size=args.test_size, random_state=42)

    move_files(train_items, "train", out_dir)
    move_files(val_items, "val", out_dir)

    class_names = sorted(list(class_to_id.keys()))
    with open(out_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.resolve()}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(class_names)}\nnames: {class_names}\n")

    with open(out_dir / "names.txt", "w", encoding="utf-8") as f:
        for name in class_names: f.write(name + "\n")

    with open(out_dir / "sources.json", 'w', encoding='utf-8') as sf:
        json.dump(sources, sf, indent=2)

    print("\nVisDrone -> YOLO conversion complete.")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Dataset written to: {out_dir.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format.")
    parser.add_argument('--ann', required=True, help="Path to the VisDrone annotations directory.")
    parser.add_argument('--images', required=True, help="Path to the VisDrone images directory.")
    parser.add_argument('--out', required=True, help="Output directory for the YOLO dataset.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Validation split fraction.")
    args = parser.parse_args()
    convert(args)