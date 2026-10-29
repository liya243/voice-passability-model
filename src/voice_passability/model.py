"""Offline HuBERT regression architecture, matching the released state dict."""
import torch
from torch import nn
from transformers import HubertConfig,HubertModel

class HubertsHeads(nn.Module):
    def __init__(self,hidden_size,num_bins=20,dropout=.1):
        super().__init__()
        self.reg_head=nn.Sequential(nn.Linear(hidden_size,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,1))
        self.cls_head=nn.Sequential(nn.Linear(hidden_size,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,num_bins))
    def forward(self,x):return self.reg_head(x).squeeze(-1),self.cls_head(x)

class HubertRegressor(nn.Module):
    def __init__(self,config: HubertConfig,num_bins=20,dropout=.1):
        super().__init__()
        self.backbone=HubertModel(config)
        self.heads=HubertsHeads(config.hidden_size,num_bins,dropout)
    def forward(self,input_values,attention_mask):
        hidden=self.backbone(input_values=input_values,attention_mask=attention_mask).last_hidden_state
        mask=self.backbone._get_feature_vector_attention_mask(hidden.shape[1],attention_mask).unsqueeze(-1)
        pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
        reg,cls=self.heads(pooled)
        return {'reg_probs':torch.sigmoid(reg),'reg_logits':reg,'cls_logits':cls}
