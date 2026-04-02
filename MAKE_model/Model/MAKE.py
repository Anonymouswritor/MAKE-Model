import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network Linear Layer."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)
        
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features * (grid_size + spline_order)))
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_((self.scale_spline * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)).permute(2, 0, 1).reshape(self.out_features, -1))

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.spline_weight)
        return base_output + spline_output


class MLPExpert(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )
    def forward(self, x):
        return self.net(x)


class SGAF(nn.Module):
    """Semantic-Guided Adaptive Fusion."""
    def __init__(self, gene_dim, img_dim, hidden_dim=128, dropout=0.3, topk=50, grid_size=3, spline_order=1):
        super().__init__()
        self.topk = topk
        self.hidden_dim = hidden_dim
        
        self.q_proj = nn.Linear(gene_dim, hidden_dim)
        self.k_proj = nn.Linear(img_dim, hidden_dim)
        self.v_proj = nn.Linear(img_dim, hidden_dim)
        
        self.gene_gate = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid()
        )
        self.preference_kan = KANLinear(gene_dim + hidden_dim, 1, grid_size=grid_size, spline_order=spline_order)
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat_img, feat_gene, mask=None):
        query = self.q_proj(feat_gene).unsqueeze(1)
        gate = self.gene_gate(feat_gene).unsqueeze(1)
        
        key = self.k_proj(feat_img)
        value_raw = self.v_proj(feat_img)
        value = value_raw * (1.0 + gate)
        
        attn_scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
            
        curr_k = min(self.topk, feat_img.size(1))
        topk_scores, topk_indices = torch.topk(attn_scores, k=curr_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)
        topk_weights = self.dropout(topk_weights)
        
        idx_expanded = topk_indices.transpose(1, 2).repeat(1, 1, value.size(-1))
        topk_values = torch.gather(value, 1, idx_expanded)
        
        gene_guided_context = torch.bmm(topk_weights, topk_values).squeeze(1)
        
        if mask is not None:
            img_summary = (feat_img * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-6)
        else:
            img_summary = feat_img.mean(1)
        img_summary = self.v_proj(img_summary)
        
        combined_feat = torch.cat([feat_gene, img_summary], dim=-1)
        alpha = torch.sigmoid(self.preference_kan(combined_feat)) 
        
        final_context = alpha * gene_guided_context + (1 - alpha) * img_summary
        
        return final_context, topk_weights, topk_indices, alpha


class MAKE(nn.Module):
    """Modality Adaptive KAN Experts (MAKE) Architecture."""
    def __init__(self, use_sbs=True, img_feat_size=768, spe_dim=19, sbs_dim=97, n_classes=2, 
                 dropout=0.3, topk=40, num_experts=2, temperature=3.0, use_kan=True, grid_size=5, spline_order=3): 
        super().__init__()
        self.use_sbs = use_sbs
        self.use_kan = use_kan
        self.hidden_dim = 64
        self.num_experts = num_experts 
        self.temperature = temperature
        self.num_gene_experts = max(1, num_experts // 2)
        self.num_img_experts = num_experts - self.num_gene_experts
        
        combined_gene_dim = spe_dim + (sbs_dim if use_sbs else 0)
        
        self.gene_query_proj = nn.Sequential(
            nn.Linear(combined_gene_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cross_attn = SGAF(
            gene_dim=self.hidden_dim, img_dim=self.hidden_dim, hidden_dim=self.hidden_dim, 
            dropout=dropout, topk=topk, grid_size=grid_size, spline_order=spline_order
        )

        router_input_dim = self.hidden_dim + combined_gene_dim
        self.router = KANLinear(router_input_dim, num_experts, grid_size=grid_size, spline_order=spline_order)
        self.experts = nn.ModuleList()
        
        for _ in range(self.num_gene_experts):
            if self.use_kan:
                self.experts.append(KANLinear(combined_gene_dim, n_classes, grid_size=grid_size, spline_order=spline_order))
            else:
                self.experts.append(MLPExpert(combined_gene_dim, n_classes, hidden_dim=64))

        for _ in range(self.num_img_experts):
            if self.use_kan:
                self.experts.append(KANLinear(self.hidden_dim, n_classes, grid_size=grid_size, spline_order=spline_order))
            else:
                self.experts.append(MLPExpert(self.hidden_dim, n_classes, hidden_dim=64))

        self.proj_gene = nn.Sequential(nn.Linear(combined_gene_dim, 128), nn.ReLU(), nn.Linear(128, 128))
        self.proj_img = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.ReLU(), nn.Linear(128, 128))
        self.pc1_regressor = nn.Linear(self.hidden_dim, 1)

    def forward(self, x_img, x_spe, x_sbs, mask=None, return_attention=False):
        if self.use_sbs: x_gene_raw = torch.cat([x_spe, x_sbs], dim=-1)
        else: x_gene_raw = x_spe
        
        q_gene = self.gene_query_proj(x_gene_raw)
        
        if x_img.dim() == 2: x_img = x_img.unsqueeze(0)
        h_img = self.img_proj(x_img)

        img_context, attn_weights, attn_indices, preference_alpha = self.cross_attn(h_img, q_gene, mask)
        
        router_input = torch.cat([img_context, x_gene_raw], dim=-1)
        router_logits = self.router(router_input)
        gating_weights = F.softmax(router_logits / self.temperature, dim=-1)
        
        final_logits = 0
        gene_expert_weights_for_loss = [] 
        for i, expert in enumerate(self.experts):
            w = gating_weights[:, i:i+1]
            if i < self.num_gene_experts:
                expert_out = expert(x_gene_raw)
                if self.use_kan:
                    w_flat = expert.base_weight.flatten()
                else:
                    w_flat = expert.net[0].weight.flatten()
                gene_expert_weights_for_loss.append(w_flat)
            else:
                expert_out = expert(img_context)
            final_logits += w * expert_out
            
        emb_gene = F.normalize(self.proj_gene(x_gene_raw), dim=1)
        emb_img = F.normalize(self.proj_img(img_context), dim=1)
        pred_pc1 = self.pc1_regressor(img_context).squeeze(1)
        
        output = {
            'logits': final_logits,
            'pred_pc1': pred_pc1,
            'routing_weights': gating_weights,
            'emb_gene': emb_gene,
            'emb_img': emb_img,
            'expert_weights': gene_expert_weights_for_loss, 
            'preference_alpha': preference_alpha
        }
        
        if return_attention:
            output['attn_weights'] = attn_weights
            output['attn_indices'] = attn_indices
            
        return output