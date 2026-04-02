import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, average_precision_score

try:

    from Model.MAKE import *
except ImportError:
    print("Error: models not found. Please ensure the model file exists.")
    exit()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

class MultiModalDataset(Dataset):
    def __init__(self, df, img_root, spe_root, sbs_root, is_train=True):
        self.df = df
        self.img_root = img_root
        self.spe_root = spe_root
        self.sbs_root = sbs_root
        self.is_train = is_train
        self.max_patches = 4000

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row['slide_id']) 
        label = int(row['label'])
        pc1 = float(row['pc1']) if 'pc1' in row else 0.0
        pc1_norm = pc1 / 10.0 
        
        try: 
            img = torch.load(os.path.join(self.img_root, slide_id + '.pt'), map_location='cpu')
            if isinstance(img, dict): img = img['features']
            if len(img.shape) == 1: img = img.unsqueeze(0)
            if img.shape[0] > self.max_patches:
                if self.is_train:
                    indices = torch.randperm(img.shape[0])[:self.max_patches]
                else:
                    indices = torch.linspace(0, img.shape[0]-1, self.max_patches).long()
                indices, _ = torch.sort(indices)
                img = img[indices]
        except: img = torch.zeros(1, 768)
        
        try: spe = torch.load(os.path.join(self.spe_root, slide_id + '.pt'), map_location='cpu')
        except: spe = torch.zeros(19)
        try: sbs = torch.load(os.path.join(self.sbs_root, slide_id + '.pt'), map_location='cpu')
        except: sbs = torch.zeros(97)
        
        return img, spe, sbs, torch.tensor(label).long(), torch.tensor(pc1_norm).float()

def collate_MIL(batch):
    imgs, spes, sbss, labels, pc1s = zip(*batch)
    lengths = [img.shape[0] for img in imgs]
    max_len = max(lengths)
    feature_dim = imgs[0].shape[1]
    
    padded_imgs = torch.zeros(len(imgs), max_len, feature_dim)
    mask = torch.zeros(len(imgs), max_len)
    
    for i, img in enumerate(imgs):
        end = lengths[i]
        padded_imgs[i, :end, :] = img
        mask[i, :end] = 1
        
    spes = torch.stack(spes, 0)
    sbss = torch.stack(sbss, 0)
    labels = torch.stack(labels, 0)
    pc1s = torch.stack(pc1s, 0) 
    
    return padded_imgs, mask, spes, sbss, labels, pc1s


class MAKE_Loss(nn.Module):
    def __init__(self, lambda_cl=0.1, lambda_pc1=1.0): 
        super().__init__()
        self.cls_loss = FocalLoss()
        self.reg_loss = nn.MSELoss()
        self.lambda_cl = lambda_cl
        self.lambda_pc1 = lambda_pc1 

    def forward(self, out, target_cls, target_pc1):
        l_cls = self.cls_loss(out['logits'], target_cls)
        l_pc1 = self.reg_loss(out['pred_pc1'], target_pc1)
        l_cl = 1 - F.cosine_similarity(out['emb_gene'], out['emb_img'], dim=1).mean()

        total_loss = l_cls + (self.lambda_pc1 * l_pc1) + (self.lambda_cl * l_cl)
        return total_loss, l_cls.item(), l_pc1.item(), l_cl.item()

# ================= Training Routine =================
def run_fold(args):
    seed_everything(args.seed)
    save_dir = os.path.join(args.save_root, "MAKE_full", f"fold_{args.fold}")
    os.makedirs(save_dir, exist_ok=True)
    
    log_path = os.path.join(save_dir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("Epoch,Total_Loss,Cls_Loss,PC1_Loss,CL_Loss,AUC,ACC,AUPR,F1\n")

    train_df = pd.read_csv(os.path.join(args.cv_dir, f"fold_{args.fold}", "train.csv"))
    val_df = pd.read_csv(os.path.join(args.cv_dir, f"fold_{args.fold}", "val.csv"))
    
    train_set = MultiModalDataset(train_df, args.img_root, args.spe_root, args.sbs_root, is_train=True)
    val_set = MultiModalDataset(val_df, args.img_root, args.spe_root, args.sbs_root, is_train=False)

    targets = train_df['label'].values
    class_weights = [1.0/(np.sum(targets==i)+1e-5) for i in range(2)]
    weights = torch.tensor([class_weights[t] for t in targets]).double()
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_set, batch_size=16, sampler=sampler, num_workers=4, collate_fn=collate_MIL)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_MIL)
   
    model = MAKE(
            num_experts=args.num_experts, 
            topk=args.topk, 
            temperature=3.0, 
            use_kan=(args.use_kan == 1),
            grid_size=args.grid_size,        
            spline_order=args.spline_order 
        ).cuda()

    criterion = MAKE_Loss(
            lambda_cl=args.w_cl, 
            lambda_pc1=args.w_pc1
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_auc = 0.0
    
    for epoch in range(args.max_epochs):
        model.train()
        losses = {'total':0, 'cls':0, 'pc1':0, 'cl':0}
        
        for img, mask, spe, sbs, label, pc1 in train_loader:
            img, mask, spe, sbs = img.cuda(), mask.cuda(), spe.cuda(), sbs.cuda()
            label, pc1 = label.cuda(), pc1.cuda()
            
            optimizer.zero_grad()
            out = model(img, spe, sbs, mask=mask)
            loss, l_c, l_p, l_cl = criterion(out, label, pc1)
            loss.backward()
            optimizer.step()
            
            losses['total'] += loss.item()
            losses['cls'] += l_c
            losses['pc1'] += l_p 
            losses['cl'] += l_cl
            
        for k in losses: losses[k] /= len(train_loader)
        
        model.eval()
        probs, labels_list = [], []
        with torch.no_grad():
            for img, mask, spe, sbs, label, _ in val_loader:
                img, mask, spe, sbs, label = img.cuda(), mask.cuda(), spe.cuda(), sbs.cuda(), label.cuda()
                out = model(img, spe, sbs, mask=mask)
                probs.append(torch.softmax(out['logits'], dim=1)[:, 1].item())
                labels_list.append(label.item())
        
        acc, auc, aupr, f1, sens, spec = calculate_metrics(probs, labels_list)
        with open(log_path, "a") as f:
            f.write(f"{epoch},{losses['total']:.4f},{losses['cls']:.4f},{losses['pc1']:.4f},{losses['cl']:.4f},"
                    f"{auc:.4f},{acc:.4f},{aupr:.4f},{f1:.4f}\n")
            
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    print(f"Fold {args.fold} Done. Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_experts', type=int, default=2)
    parser.add_argument('--w_pc1', type=float, default=0.1, help="Weight for PC1 regression loss")
    parser.add_argument('--w_cl', type=float, default=0.1, help="Weight for cl loss")
    parser.add_argument('--cv_dir', type=str, default="./data/cv_splits")
    parser.add_argument('--img_root', type=str, default="./data/features/pt_files")
    parser.add_argument('--spe_root', type=str, default="./data/features/vaf_gene_feats")
    parser.add_argument('--sbs_root', type=str, default="./data/features/sbs96_context")
    parser.add_argument('--save_root', type=str, default="./results")
    parser.add_argument('--use_kan', type=int, default=1, help="1 for KAN experts, 0 for MLP experts")
    parser.add_argument('--topk', type=int, default=50, help="Number of patches to select")
    parser.add_argument('--grid_size', type=int, default=5, help="Grid size for KAN (higher = more fine-grained)")
    parser.add_argument('--spline_order', type=int, default=3, help="Spline order for KAN (smoothness)")
    args = parser.parse_args()
    run_fold(args)