import os
import torch
import pandas as pd
import numpy as np

def create_dummy_data():
    print("Generating dummy data for codebase verification...")
    
    # Create directories
    dirs = [
        "./data/cv_splits/fold_0",
        "./data/cv_splits",
        "./data/features/pt_files",
        "./data/features/vaf_gene_feats",
        "./data/features/sbs96_context"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    # Generate dummy patient IDs
    cases = [f"patient_{i:03d}" for i in range(20)] # 20 fake patients
    
    # 1. Generate CSVs
    data = []
    for case in cases:
        label = np.random.randint(0, 2)
        pc1 = np.random.uniform(0, 10)
        data.append({"slide_id": case, "label": label, "pc1": pc1})
        
    df = pd.DataFrame(data)
    # Split into train/val/test
    df.iloc[:12].to_csv("./data/cv_splits/fold_0/train.csv", index=False)
    df.iloc[12:16].to_csv("./data/cv_splits/fold_0/val.csv", index=False)
    df.iloc[16:].to_csv("./data/cv_splits/test_set.csv", index=False)
    
    # 2. Generate dummy tensor files
    for case in cases:
        # Fake WSI features: [num_patches, 768] (random patches between 50 and 200)
        num_patches = np.random.randint(50, 200)
        wsi_feat = torch.rand(num_patches, 768)
        torch.save(wsi_feat, f"./data/features/pt_files/{case}.pt")
        
        # Fake Gene features 
        gene_feat = torch.rand(19)
        torch.save(gene_feat, f"./data/features/vaf_gene_feats/{case}.pt")
        
        # Fake SBS features (97 dim)
        sbs_feat = torch.rand(97)
        torch.save(sbs_feat, f"./data/features/sbs96_context/{case}.pt")

    print("Dummy data generated successfully in ./data/")

if __name__ == "__main__":
    create_dummy_data()