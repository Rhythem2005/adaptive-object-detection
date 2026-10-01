import os
import lance
import pyarrow as pa
import pandas as pd
import numpy as np
import yaml
from PIL import Image
import io
import time

def main():
    print("Resuming...")
    ds = lance.dataset("hf://datasets/lance-format/BDD100K-enriched/data/bdd100k.lance", storage_options={})
    
    df = ds.to_table(columns=[
        "image_id", "split", "width", "height", "weather", 
        "timeofday", "ann_categories", "ann_bboxes"
    ]).to_pandas()
    
    df['seq_id'] = df['image_id'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    df['num_boxes'] = df['ann_bboxes'].apply(len)
    df_valid = df[df['num_boxes'] > 0].copy()
    
    seq_info = df_valid.groupby('seq_id').first().reset_index()
    seq_info = seq_info.sample(frac=1, random_state=42).reset_index(drop=True)
    
    target_train = 3500
    target_val = 750
    target_test = 750
    
    selected_train_seqs = []
    selected_val_seqs = []
    selected_test_seqs = []
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    frames_per_seq = df_valid.groupby('seq_id').size().to_dict()
    
    for seq in seq_info['seq_id']:
        count = frames_per_seq[seq]
        if train_count < target_train:
            selected_train_seqs.append(seq)
            train_count += count
        elif val_count < target_val:
            selected_val_seqs.append(seq)
            val_count += count
        elif test_count < target_test:
            selected_test_seqs.append(seq)
            test_count += count
        else:
            break
            
    df_train = df_valid[df_valid['seq_id'].isin(selected_train_seqs)]
    df_val = df_valid[df_valid['seq_id'].isin(selected_val_seqs)]
    df_test = df_valid[df_valid['seq_id'].isin(selected_test_seqs)]
    
    df_train['target_split'] = 'train'
    df_val['target_split'] = 'val'
    df_test['target_split'] = 'test'
    
    df_sampled = pd.concat([df_train, df_val, df_test])
    
    class_mapping = {
        'car': 0, 'bus': 1, 'truck': 2, 'person': 3,
        'rider': 4, 'bicycle': 5, 'bike': 5, 'traffic light': 6
    }
    
    base_dir = "data/detection_dataset"
    selected_indices = df_sampled.index.tolist()
    
    batch_size = 100
    total = len(selected_indices)
    processed = 0
    
    for i in range(0, total, batch_size):
        batch_indices = selected_indices[i:i+batch_size]
        
        # Check if already processed
        # We can just check the first and last of this batch
        all_exist = True
        for b_idx in batch_indices:
            img_id = df_sampled.loc[b_idx, 'image_id']
            split = df_sampled.loc[b_idx, 'target_split']
            if not os.path.exists(os.path.join(base_dir, "images", split, f"{img_id}.jpg")):
                all_exist = False
                break
        
        if all_exist:
            processed += len(batch_indices)
            continue
            
        print(f"Resuming at {processed}/{total} images...")
        # Add a sleep to prevent rate limiting
        time.sleep(2)
        
        try:
            batch_table = ds.take(batch_indices, columns=["image_id", "image_bytes", "width", "height", "ann_categories", "ann_bboxes"])
        except Exception as e:
            print("Rate limited, sleeping for 10s:", e)
            time.sleep(10)
            batch_table = ds.take(batch_indices, columns=["image_id", "image_bytes", "width", "height", "ann_categories", "ann_bboxes"])
            
        batch_df = batch_table.to_pandas()
        
        for _, row in batch_df.iterrows():
            img_id = row['image_id']
            split = df_sampled.loc[df_sampled['image_id'] == img_id, 'target_split'].values[0]
            
            img_path = os.path.join(base_dir, "images", split, f"{img_id}.jpg")
            label_path = os.path.join(base_dir, "labels", split, f"{img_id}.txt")
            
            if not os.path.exists(img_path):
                img_bytes = row['image_bytes']
                img = Image.open(io.BytesIO(img_bytes))
                img.convert("RGB").save(img_path)
            
            if not os.path.exists(label_path):
                w, h = row['width'], row['height']
                yolo_lines = []
                cats = row['ann_categories']
                bboxes = row['ann_bboxes']
                
                if cats is not None and bboxes is not None:
                    for cat, bbox in zip(cats, bboxes):
                        if cat in class_mapping:
                            class_id = class_mapping[cat]
                            x_min, y_min, x_max, y_max = bbox
                            
                            box_w = x_max - x_min
                            box_h = y_max - y_min
                            center_x = x_min + box_w / 2.0
                            center_y = y_min + box_h / 2.0
                            
                            center_x /= w
                            center_y /= h
                            box_w /= w
                            box_h /= h
                            
                            center_x = max(0.0, min(1.0, center_x))
                            center_y = max(0.0, min(1.0, center_y))
                            box_w = max(0.0, min(1.0, box_w))
                            box_h = max(0.0, min(1.0, box_h))
                            
                            if box_w > 0 and box_h > 0:
                                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
                                
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                    if len(yolo_lines) > 0:
                        f.write("\n")
                        
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total} images...")
                
    print(f"Dataset completely saved to {base_dir}")

    print("\nSTEP 8 - QUALITY CHECK")
    errors = 0
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    with open(yaml_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
        
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_dir, "images", split)
        lbl_dir = os.path.join(base_dir, "labels", split)
        
        imgs = set([f.split('.')[0] for f in os.listdir(img_dir) if f.endswith('.jpg')])
        lbls = set([f.split('.')[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        
        if imgs != lbls:
            print(f"ERROR: Image and label mismatch in {split}!")
            errors += 1
            
        print(f"Validated {split}: {len(imgs)} images, {len(lbls)} labels")
        
        for lbl_file in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, lbl_file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"ERROR: Invalid label format in {lbl_file}")
                        errors += 1
                    else:
                        cid = int(parts[0])
                        if cid not in yaml_content['names']:
                            print(f"ERROR: Invalid class ID {cid} in {lbl_file}")
                            errors += 1
                        cx, cy, bw, bh = map(float, parts[1:])
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                            print(f"ERROR: Bbox out of bounds in {lbl_file}")
                            errors += 1
                            
    if errors == 0:
        print("\nAll quality checks passed successfully!")
    else:
        print(f"\nWARNING: Found {errors} errors during validation!")

if __name__ == "__main__":
    main()
