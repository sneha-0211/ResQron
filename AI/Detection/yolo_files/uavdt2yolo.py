import argparse
import json
import shutil
import csv
import xml.etree.ElementTree as ET
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

def convert_txt_format(ann_dir, out_dir, image_dir):
    """Handles gt.txt format for UAVDT."""
    # UAVDT class mapping
    uavdt_classes = {1: 'car', 2: 'truck', 3: 'bus'}
    class_to_id = {name: i for i, name in enumerate(uavdt_classes.values())}

    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    
    annotations_by_image = {}
    sources = {}
    
    txt_files = sorted(list(ann_dir.rglob("*.txt")))
    for txt_file in txt_files:
        sequence_name = txt_file.parent.parent.name.replace('_v', '')
        
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                
                frame_idx = int(parts[0])
                xmin = float(parts[2])
                ymin = float(parts[3])
                w_box = float(parts[4])
                h_box = float(parts[5])
                class_id_uavdt = int(parts[7])
                
                if class_id_uavdt not in uavdt_classes: continue 
                
                class_name = uavdt_classes[class_id_uavdt]
                
                # Create an image key WITHOUT the extension
                img_name_stem = f"img{frame_idx:05d}"
                img_key = f"{sequence_name}_{img_name_stem}"
                
                annotations_by_image.setdefault(img_key, []).append((xmin, ymin, w_box, h_box, class_name))
                sources[img_key] = {'dataset': 'uavdt-txt', 'source_ann': str(txt_file)}

    all_items = []
    for img_key, anns in annotations_by_image.items():
        sequence_name, img_name_stem = img_key.split('_', 1)
        
        # Robustly find the image with any common extension
        src_img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = image_dir / sequence_name / (img_name_stem + ext)
            if potential_path.exists():
                src_img_path = potential_path
                break
        
        if not src_img_path:
            print(f"Warning: Image not found for annotations in sequence '{sequence_name}' with base name '{img_name_stem}'")
            continue
            
        # Create final destination name and copy the file
        dest_img_name = f"{img_key}{src_img_path.suffix}"
        shutil.copy(src_img_path, out_img_dir / dest_img_name)
        all_items.append(dest_img_name)
        
        w, h = get_image_size(src_img_path)
        yolo_labels = []
        for xmin, ymin, w_box, h_box, class_name in anns:
            cls_id = class_to_id[class_name]
            xc = xmin + w_box / 2.0
            yc = ymin + h_box / 2.0
            yolo_labels.append([cls_id, xc / w, yc / h, w_box / w, h_box / h])
        
        # Label file is named after the key (without extension)
        write_label(out_lbl_dir / (img_key + ".txt"), yolo_labels)
        
    return class_to_id, all_items, sources

def convert_xml_format(ann_dir, out_dir, image_dir):
    """Handles VOC XML format part of UAVDT."""
    class_set, all_items, sources, temp_labels = set(), [], {}, {}
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    xml_files = sorted(list(ann_dir.rglob("*.xml")))
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
        sources[src_path.name] = {'dataset': 'uavdt-xml', 'source_ann': str(xml_file)}

    class_to_id = {name: i for i, name in enumerate(sorted(class_set))}
    for filename, labels in temp_labels.items():
        yolo_labels = []
        for l in labels:
            yolo_labels.append([class_to_id[l[0]]] + l[1:])
        write_label(out_lbl_dir / (Path(filename).stem + ".txt"), yolo_labels)
    return class_to_id, all_items, sources

def convert_csv_format(ann_dir, out_dir, image_dir):
    """Handles CSV format part of UAVDT."""
    class_set, sources, annotations_by_image = set(), {}, {}
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    csv_files = sorted(list(ann_dir.rglob("*.csv")))
    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6: continue
                frame_name, xmin, ymin, xmax, ymax, class_name = row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4]), row[5]
                annotations_by_image.setdefault(frame_name, []).append((xmin, ymin, xmax, ymax, class_name))
                class_set.add(class_name)
                sources.setdefault(frame_name, {'dataset': 'uavdt-csv', 'source_ann': str(csv_file)})
    all_items = []
    for frame_name, anns in annotations_by_image.items():
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            cand_path = image_dir / (frame_name + ext)
            if cand_path.exists(): image_path = cand_path; break
        if not image_path: continue
        shutil.copy(image_path, out_img_dir / image_path.name)
        all_items.append(image_path.name)
    class_to_id = {name: i for i, name in enumerate(sorted(class_set))}
    for item in all_items:
        frame_name = Path(item).stem
        w, h = get_image_size(out_img_dir / item)
        yolo_labels = []
        for xmin, ymin, xmax, ymax, cname in annotations_by_image.get(frame_name, []):
            xc, yc, w_box, h_box = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
            yolo_labels.append([class_to_id[cname], xc / w, yc / h, w_box / w, h_box / h])
        write_label(out_lbl_dir / (frame_name + ".txt"), yolo_labels)
    return class_to_id, all_items, sources

# --- Main Conversion Logic ---

def convert(args):
    """Main function to perform UAVDT to YOLO conversion."""
    ann_dir = Path(args.ann)
    image_dir = Path(args.images)
    out_dir = Path(args.out)
    ensure_dir(out_dir / "images")
    ensure_dir(out_dir / "labels")

    xml_files = list(ann_dir.rglob("*.xml"))
    csv_files = list(ann_dir.rglob("*.csv"))
    txt_files = list(ann_dir.rglob("*.txt"))

    if txt_files:
        print(f"Info: Found {len(txt_files)} UAVDT TXT files. Processing...")
        class_to_id, all_items, sources = convert_txt_format(ann_dir, out_dir, image_dir)
    elif xml_files:
        print(f"Info: Found {len(xml_files)} UAVDT XML files. Processing...")
        class_to_id, all_items, sources = convert_xml_format(ann_dir, out_dir, image_dir)
    elif csv_files:
        print(f"Info: Found {len(csv_files)} UAVDT CSV files. Processing...")
        class_to_id, all_items, sources = convert_csv_format(ann_dir, out_dir, image_dir)
    else:
        raise FileNotFoundError(f"No UAVDT annotations (TXT, XML, or CSV) found recursively in {ann_dir}")

    if not all_items:
        raise RuntimeError("No UAVDT images were processed. Check paths and file extensions. Ensure your folder structure is correct.")

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

    print("\nUAVDT -> YOLO conversion complete.")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Dataset written to: {out_dir.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert UAVDT annotations to YOLO format.")
    parser.add_argument('--ann', required=True, help="Path to the UAVDT annotations directory.")
    parser.add_argument('--images', required=True, help="Path to the UAVDT images directory.")
    parser.add_argument('--out', required=True, help="Output directory for the YOLO dataset.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Validation split fraction.")
    args = parser.parse_args()
    convert(args)

