import os
import yaml
import glob
import math
from pathlib import Path
from PIL import Image

def validate_dataset():
    report = []
    def log(msg=""):
        report.append(msg)

    yaml_path = "Prep-Data-Sampled/data.yaml"
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_path = data.get('path', os.getcwd())
    
    train_img_dir = data.get('train', '')
    if not os.path.isabs(train_img_dir):
        train_img_dir = os.path.join(base_path, train_img_dir)
        
    val_img_dir = data.get('val', '')
    if not os.path.isabs(val_img_dir):
        val_img_dir = os.path.join(base_path, val_img_dir)
        
    test_img_dir = data.get('test', '')
    if not os.path.isabs(test_img_dir):
        test_img_dir = os.path.join(base_path, test_img_dir)

    classes = data.get('names', {})
    if isinstance(classes, list):
        class_names = {i: name for i, name in enumerate(classes)}
    else:
        class_names = classes
    num_classes = data.get('nc', len(class_names))
    valid_class_ids = set(range(num_classes))

    # Helper to get label dir from img dir
    def get_label_dir(img_dir):
        return img_dir.replace('/images', '/labels').replace('\\images', '\\labels')
        
    splits = {
        'Train': (train_img_dir, get_label_dir(train_img_dir)),
        'Validation': (val_img_dir, get_label_dir(val_img_dir)),
        'Test': (test_img_dir, get_label_dir(test_img_dir))
    }

    metrics = {
        'Train': {'images': 0, 'labels': 0},
        'Validation': {'images': 0, 'labels': 0},
        'Test': {'images': 0, 'labels': 0},
    }
    
    total_images = 0
    total_labels = 0
    
    format_errors = []
    class_id_errors = []
    missing_labels = []
    orphan_labels = []
    invalid_boxes = []
    suspicious_annotations = []
    split_leakage = []
    yaml_consistency = []
    
    class_counts = {cid: 0 for cid in valid_class_ids}
    used_classes = set()
    
    if len(class_names) != num_classes:
        yaml_consistency.append(f"YAML consistency warning: 'nc' ({num_classes}) doesn't match 'names' length ({len(class_names)}).")
    
    image_files_by_split = {}
    label_files_by_split = {}
    
    SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    
    for split_name, (i_dir, l_dir) in splits.items():
        if not os.path.exists(i_dir):
            yaml_consistency.append(f"Split {split_name} image directory not found: {i_dir}")
            image_files_by_split[split_name] = set()
            label_files_by_split[split_name] = set()
            continue
            
        imgs = []
        for root, dirs, files in os.walk(i_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTS:
                    imgs.append(os.path.join(root, f))
        
        lbls = []
        if os.path.exists(l_dir):
            for root, dirs, files in os.walk(l_dir):
                for f in files:
                    if f.endswith('.txt'):
                        lbls.append(os.path.join(root, f))
                        
        metrics[split_name]['images'] = len(imgs)
        metrics[split_name]['labels'] = len(lbls)
        total_images += len(imgs)
        total_labels += len(lbls)
        
        image_files_by_split[split_name] = set(imgs)
        label_files_by_split[split_name] = set(lbls)
        
    all_imgs = []
    img_to_split = {}
    for split_name, imgs in image_files_by_split.items():
        for img in imgs:
            name = os.path.basename(img)
            if name in img_to_split:
                split_leakage.append(f"Image {name} appears in both {img_to_split[name]} and {split_name}.")
            else:
                img_to_split[name] = split_name
    
    for split_name, (i_dir, l_dir) in splits.items():
        imgs = image_files_by_split[split_name]
        lbls = label_files_by_split[split_name]
        
        img_basenames = {os.path.splitext(os.path.basename(i))[0]: i for i in imgs}
        lbl_basenames = {os.path.splitext(os.path.basename(l))[0]: l for l in lbls}
        
        for b_img, p_img in img_basenames.items():
            if b_img not in lbl_basenames:
                missing_labels.append(f"{p_img} has no corresponding label file.")
                
        for b_lbl, p_lbl in lbl_basenames.items():
            if b_lbl not in img_basenames:
                orphan_labels.append(f"{p_lbl} has no corresponding image file.")
                
        for lbl_path in lbls:
            b_lbl = os.path.splitext(os.path.basename(lbl_path))[0]
            
            img_path = img_basenames.get(b_lbl)
            img_w, img_h = None, None
            if img_path:
                try:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                except Exception as e:
                    format_errors.append(f"Could not read image {img_path}: {e}")
                    
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            seen_annotations = set()
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue # Ignoring empty lines as per typical YOLO format (but could flag them)
                    
                parts = line.split()
                if len(parts) != 5:
                    format_errors.append(f"{lbl_path}:{line_idx+1}: Line has {len(parts)} fields, expected exactly 5")
                    continue
                    
                try:
                    c_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    format_errors.append(f"{lbl_path}:{line_idx+1}: Non-numeric value in fields")
                    continue
                    
                if any(math.isnan(v) or math.isinf(v) for v in [x_c, y_c, w, h]):
                    format_errors.append(f"{lbl_path}:{line_idx+1}: NaN or Infinity in coordinates")
                    continue
                    
                used_classes.add(c_id)
                if c_id not in valid_class_ids:
                    class_id_errors.append(f"{lbl_path}:{line_idx+1}: Class ID {c_id} is outside declared range (0-{num_classes-1})")
                else:
                    class_counts[c_id] = class_counts.get(c_id, 0) + 1
                    
                if not (0 <= x_c <= 1):
                    format_errors.append(f"{lbl_path}:{line_idx+1}: x_center {x_c} not in [0, 1]")
                if not (0 <= y_c <= 1):
                    format_errors.append(f"{lbl_path}:{line_idx+1}: y_center {y_c} not in [0, 1]")
                if not (0 < w <= 1):
                    format_errors.append(f"{lbl_path}:{line_idx+1}: width {w} not in (0, 1]")
                if not (0 < h <= 1):
                    format_errors.append(f"{lbl_path}:{line_idx+1}: height {h} not in (0, 1]")
                    
                annot_tuple = (c_id, x_c, y_c, w, h)
                if annot_tuple in seen_annotations:
                    suspicious_annotations.append(f"{lbl_path}:{line_idx+1}: Exact duplicate annotation {annot_tuple}")
                seen_annotations.add(annot_tuple)
                
                if img_w is not None and img_h is not None:
                    abs_xc = x_c * img_w
                    abs_yc = y_c * img_h
                    abs_w = w * img_w
                    abs_h = h * img_h
                    
                    x1 = abs_xc - abs_w / 2
                    y1 = abs_yc - abs_h / 2
                    x2 = abs_xc + abs_w / 2
                    y2 = abs_yc + abs_h / 2
                    
                    if x1 < -1 or y1 < -1 or x2 > img_w + 1 or y2 > img_h + 1:
                        invalid_boxes.append(f"{lbl_path}:{line_idx+1}: Box extends outside image boundaries (img: {img_w}x{img_h}, box: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})")
                        
                    if abs_w < 2 or abs_h < 2:
                        suspicious_annotations.append(f"{lbl_path}:{line_idx+1}: Extremely tiny box (w={abs_w:.1f}px, h={abs_h:.1f}px)")
                        
                    if w > 0.98 and h > 0.98:
                        suspicious_annotations.append(f"{lbl_path}:{line_idx+1}: Box covers almost entire image (w={w:.2f}, h={h:.2f})")
                        
            if len(lines) > 100:
                suspicious_annotations.append(f"{lbl_path}: Unusually large number of objects ({len(lines)})")

    for cid in valid_class_ids:
        if class_counts.get(cid, 0) == 0:
            class_id_errors.append(f"Class ID {cid} ({class_names.get(cid, 'unknown')}) declared in YAML but never used.")

    log("DATASET VALIDATION REPORT")
    log()
    log("Dataset:")
    log(f"Images: {total_images}")
    log(f"Labels: {total_labels}")
    log()
    for split in ['Train', 'Validation', 'Test']:
        log(f"{split}:")
        log(f"  Images: {metrics[split]['images']}")
        log(f"  Labels: {metrics[split]['labels']}")
        log()
        
    log("Classes:")
    for cid in sorted(valid_class_ids):
        count = class_counts.get(cid, 0)
        name = class_names.get(cid, 'unknown')
        log(f"  {cid}: {name} — {count} annotations")
    
    for cid in sorted(list(used_classes)):
        if cid not in valid_class_ids:
            count = class_counts.get(cid, 0)
            log(f"  {cid}: UNKNOWN_CLASS — {count} annotations")
    log()
    
    def print_section(title, items, limit=50):
        log(f"{title}:")
        if not items:
            log("  None")
        else:
            for item in items[:limit]:
                log(f"  {item}")
            if len(items) > limit:
                log(f"  ... and {len(items)-limit} more")
        log()

    print_section("FORMAT ERRORS", format_errors)
    print_section("CLASS-ID ERRORS", class_id_errors)
    print_section("MISSING LABELS", missing_labels)
    print_section("ORPHAN LABELS", orphan_labels)
    print_section("INVALID BOUNDING BOXES", invalid_boxes)
    print_section("DUPLICATE/SUSPICIOUS ANNOTATIONS", suspicious_annotations)
    print_section("SPLIT LEAKAGE", split_leakage)
    print_section("YAML CONSISTENCY", yaml_consistency)
    
    has_errors = any([format_errors, class_id_errors, invalid_boxes, split_leakage, yaml_consistency])
    has_warnings = any([missing_labels, orphan_labels, suspicious_annotations])
    
    if has_errors:
        status = "FAIL"
    elif has_warnings:
        status = "PASS WITH WARNINGS"
    else:
        status = "PASS"
        
    log("FINAL STATUS:")
    log(f"  {status}")
    log()
    log("**Are these labels safe and structurally valid for YOLOv8 fine-tuning?**")
    if status == "PASS":
        log("YES")
    elif status == "PASS WITH WARNINGS":
        log("PASS WITH WARNINGS - Structurally valid, but consider checking missing labels or suspicious boxes.")
    else:
        log("NO - The dataset contains structural errors that will likely crash YOLOv8 training or cause invalid metrics.")
        
    with open("validation_report.txt", "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    validate_dataset()
