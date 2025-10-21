import argparse
import json
import shutil
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

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

# --- Conversion Functions ---

def convert_coco(json_path: Path, out_dir: Path, image_dir: Path):
    with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    class_names = sorted(list(set(categories.values())))
    class_to_id = {name: i for i, name in enumerate(class_names)}
    ann_by_img = {}
    for ann in data["annotations"]: ann_by_img.setdefault(ann["image_id"], []).append(ann)
    all_items, sources = [], {}
    out_img_dir, out_lbl_dir = out_dir / "images", out_dir / "labels"
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
        sources[file_name] = {'dataset': 'coco', 'source_ann': str(json_path)}
    return class_to_id, all_items, sources

def convert_voc(xml_dir: Path, out_dir: Path, image_dir: Path):
    class_set, all_items, sources, temp_labels = set(), [], {}, {}
    out_img_dir, out_lbl_dir = out_dir / "images", out_dir / "labels"
    xml_files = sorted(list(xml_dir.rglob("*.xml")))
    if not xml_files: raise FileNotFoundError(f"No XML files found in {xml_dir}")
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text.strip()
        src_path = image_dir / Path(filename).name
        if not src_path.exists(): continue
        shutil.copy(src_path, out_img_dir / src_path.name)
        size = root.find("size")
        w, h = (float(size.find("width").text), float(size.find("height").text)) if size else get_image_size(src_path)
        if w == 0 or h == 0: continue
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            class_set.add(name)
            bnd = obj.find("bndbox")
            xmin, ymin, xmax, ymax = map(float, [bnd.find(t).text for t in ['xmin', 'ymin', 'xmax', 'ymax']])
            xc, yc, w_box, h_box = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
            labels.append([name, xc / w, yc / h, w_box / w, h_box / h])
        temp_labels[src_path.name] = labels
        all_items.append(src_path.name)
        sources[src_path.name] = {'dataset': 'voc', 'source_ann': str(xml_file)}
    class_to_id = {name: i for i, name in enumerate(sorted(class_set))}
    for filename, labels in temp_labels.items():
        yolo_labels = [[class_to_id[l[0]]] + l[1:] for l in labels]
        write_label(out_lbl_dir / (Path(filename).stem + ".txt"), yolo_labels)
    return class_to_id, all_items, sources

def convert_csv(csv_file: Path, out_dir: Path):
    class_set, sources, annotations_by_image = set(), {}, {}
    out_img_dir, out_lbl_dir = out_dir / "images", out_dir / "labels"
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 6: continue
            img_path_str, xmin, ymin, xmax, ymax, class_name = row
            img_path = Path(img_path_str)
            if not img_path.exists(): continue
            shutil.copy(img_path, out_img_dir / img_path.name)
            class_set.add(class_name)
            annotations_by_image.setdefault(img_path.name, []).append(
                (float(xmin), float(ymin), float(xmax), float(ymax), class_name)
            )
    class_to_id = {name: i for i, name in enumerate(sorted(class_set))}
    all_items = list(annotations_by_image.keys())
    sources = {item: {'dataset': 'csv', 'source_ann': str(csv_file)} for item in all_items}
    for file_name, anns in annotations_by_image.items():
        yolo_labels = []
        w, h = get_image_size(out_img_dir / file_name)
        for xmin, ymin, xmax, ymax, cname in anns:
            xc, yc, w_box, h_box = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
            yolo_labels.append([class_to_id[cname], xc / w, yc / h, w_box / w, h_box / h])
        write_label(out_lbl_dir / (Path(file_name).stem + ".txt"), yolo_labels)
    return class_to_id, all_items, sources

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Convert COCO, VOC, or CSV annotations to YOLOv8 format.")
    parser.add_argument("--format", required=True, choices=["coco", "voc", "csv"], help="Input dataset format.")
    parser.add_argument("--ann", required=True, type=Path, help="Path to annotations (file or directory).")
    parser.add_argument("--images", type=Path, help="Path to images directory (for coco/voc).")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the YOLO dataset.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of the dataset for validation.")
    args = parser.parse_args()

    ensure_dir(args.out / "images")
    ensure_dir(args.out / "labels")

    if args.format == "coco":
        if not args.images: raise ValueError("--images path is required for COCO format.")
        class_to_id, all_items, sources = convert_coco(args.ann, args.out, args.images)
    elif args.format == "voc":
        if not args.images: raise ValueError("--images path is required for VOC format.")
        class_to_id, all_items, sources = convert_voc(args.ann, args.out, args.images)
    elif args.format == "csv":
        class_to_id, all_items, sources = convert_csv(args.ann, args.out)
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    if not all_items:
        print("Error: No items were processed. Please check your dataset paths and format.")
        return

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
    with open(args.out / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(f"path: {args.out.resolve()}\n")
        f.write("train: images/train\nval: images/val\n")
        f.write(f"nc: {len(class_names)}\nnames: {class_names}\n")

    with open(args.out / "names.txt", "w", encoding="utf-8") as f:
        for name in class_names: f.write(name + "\n")

    with open(args.out / "sources.json", 'w', encoding='utf-8') as sf:
        json.dump(sources, sf, indent=2)

    print("\nConversion complete.")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Dataset written to: {args.out.resolve()}")

if __name__ == "__main__":
    main()
