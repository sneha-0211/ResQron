import argparse
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm

# --- Configuration ---

# Define the final, unified class taxonomy for your model.
# The order here determines the final class IDs (0, 1, 2, ...).
UNIFIED_CLASSES = [
    "person",
    "vehicle"
    # NOTE: You can add more classes like "tent" or "debris" here in the future.
    # The script will map to them if they are defined in the MAPPING below.
]

CLASS_MAPPING = {
    # VisDrone Classes
    "pedestrian": "person",
    "people": "person",
    "car": "vehicle",
    "van": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "motor": "vehicle",
    "bicycle": "vehicle",
    "awning-tricycle": "vehicle",
    "tricycle": "vehicle",
    
    # DOTA Classes
    "small-vehicle": "vehicle",
    "large-vehicle": "vehicle",
    "plane": "vehicle",
    "ship": "vehicle",
    "helicopter": "vehicle",
    "container-crane": "vehicle" 

# --- Main Script Logic ---

def unify_datasets(dataset_paths, out_dir):
    """
    Merges multiple YOLO datasets into a single dataset with a unified class map.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        print(f"Output directory {out_dir} already exists. Removing it.")
        shutil.rmtree(out_dir)

    # Create the new YOLO directory structure
    out_images_train = out_dir / "images/train"
    out_images_val = out_dir / "images/val"
    out_labels_train = out_dir / "labels/train"
    out_labels_val = out_dir / "labels/val"

    for p in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"Unified classes: {UNIFIED_CLASSES}")
    unified_class_to_id = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

    # Process each dataset
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.name
        print(f"\nProcessing dataset: {dataset_name}")

        # Load the original dataset's YAML
        try:
            with open(dataset_path / "dataset.yaml", "r") as f:
                original_yaml = yaml.safe_load(f)
            original_classes = original_yaml['names']
        except Exception as e:
            print(f"Error loading yaml for {dataset_name}: {e}. Skipping.")
            continue

        # Process train and val splits
        for split in ["train", "val"]:
            image_dir = dataset_path / f"images/{split}"
            label_dir = dataset_path / f"labels/{split}"

            if not image_dir.exists():
                print(f"  - No '{split}' split found for {dataset_name}. Skipping.")
                continue

            image_files = sorted(list(image_dir.glob("*")))
            
            print(f"  - Merging {len(image_files)} images from '{split}' split...")
            for img_file in tqdm(image_files, unit="files"):
                # Create a unique new name to avoid collisions
                new_stem = f"{dataset_name}_{img_file.stem}"
                
                # Copy image
                shutil.copy(img_file, (out_dir / f"images/{split}" / f"{new_stem}{img_file.suffix}"))

                # Find and process label file
                label_file = label_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    new_labels = []
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts: continue
                            
                            original_class_id = int(parts[0])
                            original_class_name = original_classes[original_class_id]

                            if original_class_name in CLASS_MAPPING:
                                unified_class_name = CLASS_MAPPING[original_class_name]
                                if unified_class_name in unified_class_to_id:
                                    new_class_id = unified_class_to_id[unified_class_name]
                                    new_line = f"{new_class_id} {' '.join(parts[1:])}"
                                    new_labels.append(new_line)
                    
                    if new_labels:
                        with open(out_dir / f"labels/{split}" / f"{new_stem}.txt", "w") as f:
                            f.write("\n".join(new_labels))

    # Create the final dataset.yaml for the merged dataset
    final_yaml_path = out_dir / "dataset.yaml"
    final_yaml_data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES
    }
    with open(final_yaml_path, "w") as f:
        yaml.dump(final_yaml_data, f, sort_keys=False)

    print(f"\nUnification complete! Merged dataset is ready at: {out_dir.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unify multiple YOLO datasets into one.")
    parser.add_argument('--datasets', nargs='+', required=True, help="List of paths to the YOLO dataset directories to merge.")
    parser.add_argument('--out', required=True, help="Output directory for the final merged dataset.")
    args = parser.parse_args()
    
    unify_datasets(args.datasets, args.out)