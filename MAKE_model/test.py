import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, average_precision_score

try:
    from Model.MAKE import * 
except ImportError:
    print("Error: models not found. Please ensure the model file exists.")
    exit()


def calculate_metrics(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(labels, preds)
    try: auc = roc_auc_score(labels, probs)
    except: auc = 0.5
    try: aupr = average_precision_score(labels, probs)
    except: aupr = 0.0
    f1 = f1_score(labels, preds, average='macro')
    
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
    except: sens, spec = 0, 0
    
    return acc, auc, aupr, f1, sens, spec


class TestDataset(Dataset):
    def __init__(self, df, img_root, spe_root, sbs_root, max_patches=4000):
        self.df = df
        self.img_root = img_root
        self.spe_root = spe_root
        self.sbs_root = sbs_root
        self.max_patches = max_patches

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row['slide_id']) 
        label = int(row['label'])
        
        # Load Image
        try: 
            img = torch.load(os.path.join(self.img_root, slide_id + '.pt'), map_location='cpu')
            if isinstance(img, dict): img = img['features']
            if len(img.shape) == 1: img = img.unsqueeze(0)
            
            if img.shape[0] > self.max_patches:
                indices = torch.linspace(0, img.shape[0]-1, self.max_patches).long()
                img = img[indices]
                
        except: img = torch.zeros(1, 768)
        
        # Load Genes
        try: spe = torch.load(os.path.join(self.spe_root, slide_id + '.pt'), map_location='cpu')
        except: spe = torch.zeros(19)
        try: sbs = torch.load(os.path.join(self.sbs_root, slide_id + '.pt'), map_location='cpu')
        except: sbs = torch.zeros(97)
        
        return img, spe, sbs, torch.tensor(label).long()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--num_experts', type=int, default=2)
    parser.add_argument('--use_kan', type=int, default=1, help="1 for KAN experts, 0 for MLP experts")
    parser.add_argument('--topk', type=int, default=50, help="Number of patches to select")
    parser.add_argument('--grid_size', type=int, default=5, help="Grid size for KAN")
    parser.add_argument('--spline_order', type=int, default=3, help="Spline order for KAN")
    parser.add_argument('--cv_dir', type=str, default="./data/cv_splits")
    parser.add_argument('--img_root', type=str, default="./data/features/pt_files")
    parser.add_argument('--spe_root', type=str, default="./data/features/vaf_gene_feats")
    parser.add_argument('--sbs_root', type=str, default="./data/features/sbs96_context")
    parser.add_argument('--weights_dir', type=str, default="./results/MAKE_full")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    print(f"Testing MAKE on Test Set...")
    print(f"Weights Dir: {args.weights_dir}")
    
    test_df = pd.read_csv(os.path.join(args.cv_dir, "test_set.csv"))
    test_set = TestDataset(test_df, args.img_root, args.spe_root, args.sbs_root)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    
    results = []
    
    for fold in range(5):
        ckpt_path = os.path.join(args.weights_dir, f"fold_{fold}", "best_model.pth")
        
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Fold {fold} missing at {ckpt_path}, skipping.")
            continue
  
        model = MAKE(
                num_experts=args.num_experts, 
                topk=args.topk,
                use_kan=(args.use_kan == 1),
                grid_size=args.grid_size,        
                spline_order=args.spline_order 
            ).to(device)
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        probs, labels_list = [], []
        with torch.no_grad():
            for img, spe, sbs, label in test_loader:
                img, spe, sbs = img.to(device), spe.to(device), sbs.to(device)
                out = model(img, spe, sbs, mask=None)        
                probs.extend(torch.softmax(out['logits'], dim=1)[:, 1].cpu().numpy())
                labels_list.extend(label.numpy())
        acc, auc, aupr, f1, sens, spec = calculate_metrics(probs, labels_list)
        print(f"  Fold {fold}: AUC={auc:.4f} | ACC={acc:.4f} | F1={f1:.4f} | AUPR={aupr:.4f}")
        
        results.append({
            "Fold": fold, 
            "ACC": acc, "AUC": auc, "AUPR": aupr, "F1": f1
        })
        
    if results:
        df_res = pd.DataFrame(results)
        mean_metrics = df_res.iloc[:, 1:].mean()
        std_metrics = df_res.iloc[:, 1:].std()
        
        print("-" * 30)
        print(f"Final MAKE Results (5-Fold Mean):")
        print(f"AUC:  {mean_metrics['AUC']:.4f} ± {std_metrics['AUC']:.4f}")
        print(f"F1:   {mean_metrics['F1']:.4f} ± {std_metrics['F1']:.4f}")
        print(f"AUPR: {mean_metrics['AUPR']:.4f} ± {std_metrics['AUPR']:.4f}")
        print("-" * 30)
        
        save_path = os.path.join(args.weights_dir, "results_test.csv")
        df_res.loc['mean'] = df_res.mean()
        df_res.loc['std'] = df_res.std()
        df_res.to_csv(save_path, index=False)
        print(f"Saved results to: {save_path}")

if __name__ == "__main__":
    main()