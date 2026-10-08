import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

def create_stratified_split(data_root="data/BraTS2018/MICCAI_BraTS_2018_Data_Training", output_dir="datalist/BraTS18", seed=42):
    os.makedirs(output_dir, exist_ok=True)
    
    hgg_dirs = sorted([d for d in glob.glob(os.path.join(data_root, "HGG", "*")) if os.path.isdir(d)])
    lgg_dirs = sorted([d for d in glob.glob(os.path.join(data_root, "LGG", "*")) if os.path.isdir(d)])
    
    print(f"Found {len(hgg_dirs)} HGG cases and {len(lgg_dirs)} LGG cases. Total: {len(hgg_dirs) + len(lgg_dirs)}")
    
    all_paths = hgg_dirs + lgg_dirs
    all_labels = [1] * len(hgg_dirs) + [0] * len(lgg_dirs) # 1: HGG, 0: LGG
    
    # Normalize paths to use forward slashes: data/BraTS2018/MICCAI_BraTS_2018_Data_Training/HGG/...
    all_paths = [p.replace("\\", "/") for p in all_paths]
    
    # Stratified split: 80% (train + val) vs 20% test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, random_state=seed, stratify=all_labels
    )
    
    # Stratified split from train_val: out of 80%, take 1/8 (10% of total) for meta-val, remaining (70% of total) for meta-train
    # 0.1 / 0.8 = 0.125
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.125, random_state=seed, stratify=train_val_labels
    )
    
    # Verify disjointness (No Data Leakage)
    s_train = set(train_paths)
    s_val = set(val_paths)
    s_test = set(test_paths)
    
    assert len(s_train.intersection(s_val)) == 0, "Leakage between Train and Val!"
    assert len(s_train.intersection(s_test)) == 0, "Leakage between Train and Test!"
    assert len(s_val.intersection(s_test)) == 0, "Leakage between Val and Test!"
    assert len(s_train) + len(s_val) + len(s_test) == len(all_paths), "Total case count mismatch!"
    
    print(f"\n--- Split Summary (Seed={seed}) ---")
    print(f"Meta-Train: {len(train_paths)} cases ({sum(train_labels)} HGG, {len(train_labels)-sum(train_labels)} LGG)")
    print(f"Meta-Val:   {len(val_paths)} cases ({sum(val_labels)} HGG, {len(val_labels)-sum(val_labels)} LGG)")
    print(f"Local Test: {len(test_paths)} cases ({sum(test_labels)} HGG, {len(test_labels)-sum(test_labels)} LGG)")
    print(f"Total:      {len(all_paths)} cases")
    
    # Save CSVs
    def save_csv(filename, paths):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            for p in sorted(paths):
                f.write(f"{p}\n")
        print(f"Saved: {filepath} ({len(paths)} lines)")
        
    save_csv("BraTS18_metaTrain.csv", train_paths)
    save_csv("BraTS18_metaVal.csv", val_paths)
    save_csv("BraTS18_test.csv", test_paths)
    save_csv("BraTS18_train_all.csv", all_paths)
    print("Stratified Split completed successfully without any data leakage!")

if __name__ == "__main__":
    create_stratified_split()
