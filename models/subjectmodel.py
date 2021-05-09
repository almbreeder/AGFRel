import sys
sys.path.append("../")

import config
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttentionLayer

class SubJectModel(nn.Module):
    
    def __init__(self):
        
        super(SubJectModel, self).__init__()

        self.attention_heads = MultiHeadAttentionLayer(config.bert_feature_size, 12, 0.4)
        self.attention_tails = MultiHeadAttentionLayer(config.bert_feature_size, 12, 0.4)

        self.projection_heads = nn.Linear(config.bert_feature_size, 1)
        self.projection_tails = nn.Linear(config.bert_feature_size, 1)
        self.projection_heads.to(config.device)
        self.projection_tails.to(config.device)
        
    def forward(self, bert_outputs,attention_mask):

        heads_outputs, _ = self.attention_heads(bert_outputs,bert_outputs,bert_outputs,attention_mask)
        tails_outputs, _ = self.attention_heads(bert_outputs, bert_outputs, bert_outputs,attention_mask)

        #[10,200,1]
        sub_heads = self.projection_heads(heads_outputs)
        sub_tails = self.projection_tails(tails_outputs)
        
        b, _, _ = list(sub_heads.size())
        #[batch size, 200]
        sub_heads = torch.sigmoid(sub_heads).view(b, -1)
        sub_tails = torch.sigmoid(sub_tails).view(b, -1)
        
        return sub_heads, sub_tails
        
        
        
        
        