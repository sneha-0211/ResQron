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

# --- Main Conversion Logic ---

def convert(args):
    """Main function to perform DOTA to YOLO conversion."""
    ann_dir = Path(args.ann)
    image_dir = Path(args.images)
    out_dir = Path(args.out)

    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    class_set = set()
    all_items = []
    sources = {}
    temp_labels = {}

    txt_files = sorted(list(ann_dir.rglob("*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No DOTA .txt annotation files found recursively in {ann_dir}")

    print(f"Found {len(txt_files)} DOTA annotation files. Processing...")
    for txt_file in txt_files:
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            cand_path = image_dir / (txt_file.stem + ext)
            if cand_path.exists():
                image_path = cand_path
                break

        if not image_path:
            print(f"Warning: Skipping DOTA annotation, no matching image for {txt_file.name}")
            continue

        shutil.copy(image_path, out_img_dir / image_path.name)
        w, h = get_image_size(image_path)

        labels = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) < 9: continue
                coords = list(map(float, parts[0:8]))
                cls = parts[8]
                xs, ys = coords[0::2], coords[1::2]
                xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

                if xmax <= xmin or ymax <= ymin: continue
                xc, yc = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                ww, hh = xmax - xmin, ymax - ymin
                if ww < 4 or hh < 4: continue

                labels.append((cls, xc / w, yc / h, ww / w, hh / h))
                class_set.add(cls)

        temp_labels[image_path.name] = labels
        all_items.append(image_path.name)
        sources[image_path.name] = {'dataset': 'dota', 'source_ann': str(txt_file)}

    if not all_items:
        raise RuntimeError("No DOTA images were processed. Check paths and file extensions.")

    class_to_id = {name: i for i, name in enumerate(sorted(list(class_set)))}
    for filename, labels in temp_labels.items():
        yolo_labels = []
        for cls, xc, yc, w_box, h_box in labels:
            yolo_labels.append([class_to_id[cls], xc, yc, w_box, h_box])
        lbl_path = out_lbl_dir / (Path(filename).stem + ".txt")
        write_label(lbl_path, yolo_labels)

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

    print("\nDOTA -> YOLO conversion complete.")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Dataset written to: {out_dir.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DOTA annotations to YOLO format.")
    parser.add_argument('--ann', required=True, help="Path to the DOTA labelTxt directory.")
    parser.add_argument('--images', required=True, help="Path to the DOTA images directory.")
    parser.add_argument('--out', required=True, help="Output directory for the YOLO dataset.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Validation split fraction.")
    args = parser.parse_args()
    convert(args)