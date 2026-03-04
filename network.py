import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, max_h=24, max_w=24):
        super().__init__()
        self.dim = dim
        self.pos_emb_h = nn.Parameter(torch.empty(max_h, dim // 2))
        self.pos_emb_w = nn.Parameter(torch.empty(max_w, dim // 2))
        nn.init.normal_(self.pos_emb_h, std=0.02)
        nn.init.normal_(self.pos_emb_w, std=0.02)

    def forward(self, h, w):
        pos_h = self.pos_emb_h[:h].unsqueeze(1).expand(-1, w, -1)
        pos_w = self.pos_emb_w[:w].unsqueeze(0).expand(h, -1, -1)
        pos = torch.cat([pos_h, pos_w], dim=-1)
        return pos.flatten(0, 1)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class StyleContentModalityLearner(nn.Module):

    def __init__(self, dim_shallow, dim_deep, hidden_dim=768):
        super().__init__()
        self.style_compress = nn.Sequential(
            nn.Linear(dim_shallow * 2, dim_shallow),
            nn.LayerNorm(dim_shallow),
            nn.GELU()
        )
        self.content_compress = nn.Sequential(
            nn.Linear(dim_deep, dim_deep // 2),
            nn.LayerNorm(dim_deep // 2),
            nn.GELU()
        )
        fusion_dim = dim_shallow + (dim_deep // 2)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, hidden_dim)
        )

    def calc_style_stats(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        mu = x_flat.mean(dim=2) 
        std = (x_flat.var(dim=2) + 1e-6).sqrt() 
        return torch.cat([mu, std], dim=1) 

    def forward(self, feat_shallow, feat_deep):

        raw_style = self.calc_style_stats(feat_shallow)
        style_vec = self.style_compress(raw_style)     
        raw_content = F.adaptive_avg_pool2d(feat_deep, (1, 1)).flatten(1)
        content_vec = self.content_compress(raw_content) 
        concat_feat = torch.cat([style_vec, content_vec], dim=1)
        modality_vector = self.fusion_mlp(concat_feat) 
        return modality_vector


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        dim_head = dim // heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        q = self.to_q(query)
        k = self.to_k(key_value)
        v = self.to_v(key_value)
        q, k, v = map(lambda t: rearrange(t, 'n b (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> n b (h d)')
        return self.to_out(out)

class BidirectionalCrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8, hidden_dim=None, dropout=0.1, max_h=24, max_w=24):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim else dim * 4
        self.token_norm = nn.LayerNorm(dim)
        self.token_to_image_attn = CrossAttention(dim, heads=heads, dropout=dropout)
        
        self.image_norm = nn.LayerNorm(dim)
        self.image_to_token_attn = CrossAttention(dim, heads=heads, dropout=dropout)
        
        self.token_ffn_norm = nn.LayerNorm(dim)
        self.token_ffn = FeedForward(dim, hidden_dim=hidden_dim, dropout=dropout)
        
        self.image_ffn_norm = nn.LayerNorm(dim)
        self.image_ffn = FeedForward(dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, tokens, image_features):
        tokens_updated = tokens + self.token_to_image_attn(query=self.token_norm(tokens), key_value=image_features)
        tokens_updated = tokens_updated + self.token_ffn(self.token_ffn_norm(tokens_updated))
        
        image_updated = image_features + self.image_to_token_attn(query=self.image_norm(image_features), key_value=tokens)
        image_updated = image_updated + self.image_ffn(self.image_ffn_norm(image_updated))
        return tokens_updated, image_updated

class TokenImageTransformer(nn.Module):
    def __init__(self, dim=584, num_layers=2, heads=8, dropout=0.1, use_pos_encoding=True, max_h=24, max_w=24):
        super().__init__()
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding2D(dim, max_h=max_h, max_w=max_w)
        else:
            self.pos_encoding = None
            
        self.layers = nn.ModuleList([
            BidirectionalCrossAttentionLayer(dim, heads=heads, dropout=dropout, max_h=max_h, max_w=max_w)
            for _ in range(num_layers)
        ])

    def forward(self, tokens, image_features, h=None, w=None):
        if self.pos_encoding is not None and h is not None and w is not None:
            pos_emb = self.pos_encoding(h, w)
            pos_emb = pos_emb.unsqueeze(1).expand(-1, image_features.shape[1], -1)
            image_features = image_features + pos_emb

        updated_tokens = tokens
        updated_image_features = image_features
        for layer in self.layers:
            updated_tokens, updated_image_features = layer(updated_tokens, updated_image_features)
        return updated_tokens, updated_image_features

class C2P(nn.Module):
    def __init__(self, model='base', 
                 num_geo_tokens=9,        
                 num_semantic_tokens=9,   
                 num_datasets=8,          
                 use_token_supervision=True,
                 backbone_pretrained=True,
                 ):
        super().__init__()
        self.use_token_supervision = use_token_supervision
        self.bkbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', features_only=True, pretrained=backbone_pretrained)

        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512+512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(256+256, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(128+128, 64, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)

        self.num_geo_tokens = num_geo_tokens
        self.num_semantic_tokens = num_semantic_tokens
        token_dim = 584 
        hidden_dim = 768 
        
        self.geo_tokens = nn.Parameter(torch.randn(num_geo_tokens, hidden_dim))
        nn.init.xavier_uniform_(self.geo_tokens)
        
        self.semantic_tokens = nn.Parameter(torch.randn(num_semantic_tokens, hidden_dim))
        nn.init.xavier_uniform_(self.semantic_tokens)
        
        self.modality_learner = StyleContentModalityLearner(
            dim_shallow=128,  
            dim_deep=1024,    
            hidden_dim=768   
        )
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU()
        )

        self.ref_proj = nn.Sequential(nn.Linear(1024, token_dim), nn.LayerNorm(token_dim))
        self.updated_feat_proj = nn.Conv2d(token_dim, 1024, kernel_size=1, bias=False)
        
        self.d2_proj = nn.Sequential(
            nn.Conv2d(64, token_dim, kernel_size=1),
            nn.GroupNorm(num_groups=1, num_channels=token_dim) 
        )

        self.transformer_deep = TokenImageTransformer(dim=token_dim, num_layers=3, heads=8, max_h=24, max_w=24)

        geo_out_dims = [4, 1, 1, 1, 1, 2, 1, 1, 1] 
        self.geo_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, 256), nn.LayerNorm(256), nn.GELU(),
                nn.Linear(256, out_dim)
            ) for out_dim in geo_out_dims
        ])

        self.semantic_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, token_dim), nn.LayerNorm(token_dim), nn.GELU(),
                nn.Linear(token_dim, hidden_dim) 
            ) for _ in range(num_semantic_tokens)
        ])

        self.kernel_generator_fg = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim * 2), 
            nn.LayerNorm(token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim)
        )
        self.kernel_generator_bg = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim * 2),
            nn.LayerNorm(token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim)
        )
        self.cross_attn_geo = CrossAttention(dim=token_dim, heads=4, dropout=0.1)
        self.cross_attn_sem = CrossAttention(dim=token_dim, heads=4, dropout=0.1)

    def forward(self, x):

        input_img = x
        B = x.shape[0]
        E2, E3, E4, E5 = self.bkbone(x)
        _, _, H_E5, W_E5 = E5.shape
        geo_tokens = self.geo_tokens.unsqueeze(0).expand(B, -1, -1)
        if self.training:
            geo_tokens = geo_tokens + torch.randn_like(geo_tokens) * 0.01
        sem_tokens = self.semantic_tokens.unsqueeze(0).expand(B, -1, -1)
        
        modality_vec = self.modality_learner(feat_shallow=E2, feat_deep=E5) # [B, 768]

        sem_tokens = sem_tokens + modality_vec.unsqueeze(1)
        geo_tokens_proj = self.token_proj(geo_tokens)
        sem_tokens_proj = self.token_proj(sem_tokens)
        
        all_tokens = torch.cat([geo_tokens_proj, sem_tokens_proj], dim=1)
        
        img_feat_E5_flat = E5.flatten(2).permute(0, 2, 1) 
        img_feat_E5_flat = self.ref_proj(img_feat_E5_flat) 
        H_E5, W_E5 = E5.shape[2], E5.shape[3]
        
        all_tokens_T = all_tokens.permute(1, 0, 2) 
        img_feat_E5_T = img_feat_E5_flat.permute(1, 0, 2) 


        updated_tokens_L5, updated_img_L5 = self.transformer_deep(
            all_tokens_T, img_feat_E5_T, h=H_E5, w=W_E5
        )

        updated_img_L5_2d = updated_img_L5.permute(1, 2, 0).view(B, 584, H_E5, W_E5)
        proj_E5 = self.updated_feat_proj(updated_img_L5_2d)


        D5 = self.upconv5(proj_E5)
        D5 = torch.cat([D5, E4], dim=1)
        D5 = F.relu(self.bn5_1(self.conv5_1(D5)))
 
        D4 = self.upconv4(D5)
        D4 = torch.cat([D4, E3], dim=1)
        D4 = F.relu(self.bn4_1(self.conv4_1(D4)))


        D3 = self.upconv3(D4)
        D3 = torch.cat([D3, E2], dim=1)
        D2 = F.relu(self.bn3_1(self.conv3_1(D3)))
        
        D2_proj_2d = self.d2_proj(D2) 

        final_tokens = updated_tokens_L5.permute(1, 0, 2) 
        
        geo_final = final_tokens[:, :self.num_geo_tokens, :] 
        sem_final = final_tokens[:, self.num_geo_tokens:, :] 
        
        geo_predictions = {}
        if self.use_token_supervision:
            for i, predictor in enumerate(self.geo_predictors):
                key_map = ['bbox', 'area', 'perimeter', 'aspect_ratio', 'compactness', 
                           'centroid', 'eccentricity', 'orientation', 'solidity']
                geo_predictions[key_map[i]] = torch.sigmoid(predictor(geo_final[:, i]))

        semantic_embeddings = None
        if self.use_token_supervision:
            sem_preds = []
            for i in range(self.num_semantic_tokens):
                token_feat = sem_final[:, i, :]
                pred = self.semantic_projectors[i](token_feat)
                sem_preds.append(pred)
            
            semantic_embeddings = torch.stack(sem_preds, dim=1)

        img_avg = F.adaptive_avg_pool2d(D2_proj_2d, 1).flatten(1)
        image_global = img_avg
        q_feat = image_global.unsqueeze(0)
        
        k_geo = geo_final.permute(1, 0, 2)
        geo_feat = self.cross_attn_geo(query=q_feat, key_value=k_geo).squeeze(0)
        
        k_sem = sem_final.permute(1, 0, 2)
        sem_feat = self.cross_attn_sem(query=q_feat, key_value=k_sem).squeeze(0)
        
        combined_input = torch.cat([geo_feat, sem_feat, image_global], dim=1)
        fg_kernel_params = self.kernel_generator_fg(combined_input) 
        bg_kernel_params = self.kernel_generator_bg(combined_input)
        
        dk_fg = fg_kernel_params.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        dk_bg = bg_kernel_params.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        
        output_fpn = []
        output_bkg = []
        
        for bs in range(B):
            k_fg = dk_fg[bs]
            w1_fg = k_fg[:, 0:512].reshape(8, 64, 1, 1)
            w2_fg = k_fg[:, 512:576].reshape(8, 8, 1, 1)
            w3_fg = k_fg[:, 576:].reshape(1, 8, 1, 1)
            
            f = D2[bs].unsqueeze(0)
            
            out_fg = F.conv2d(f, w1_fg, stride=1, padding=0)
            out_fg = F.relu(out_fg)
            out_fg = F.conv2d(out_fg, w2_fg, stride=1, padding=0)
            out_fg = F.relu(out_fg)
            out_fg = F.conv2d(out_fg, w3_fg, stride=1, padding=0)
            
            output_fpn.append(out_fg)
            
            k_bg = dk_bg[bs]
            w1_bg = k_bg[:, 0:512].reshape(8, 64, 1, 1)
            w2_bg = k_bg[:, 512:576].reshape(8, 8, 1, 1)
            w3_bg = k_bg[:, 576:].reshape(1, 8, 1, 1)
            
            out_bg = F.conv2d(f, w1_bg, stride=1, padding=0)
            out_bg = F.relu(out_bg)
            out_bg = F.conv2d(out_bg, w2_bg, stride=1, padding=0)
            out_bg = F.relu(out_bg)
            out_bg = F.conv2d(out_bg, w3_bg, stride=1, padding=0)
            
            output_bkg.append(out_bg)

        output_fpn = torch.cat(output_fpn, dim=0)
        output_bkg = torch.cat(output_bkg, dim=0)

        output_prior = F.interpolate(output_fpn, size=input_img.size()[2:], mode='bilinear', align_corners=True)
        output_priorb = F.interpolate(output_bkg, size=input_img.size()[2:], mode='bilinear', align_corners=True)
  

        return output_prior, output_priorb, geo_predictions, semantic_embeddings
       