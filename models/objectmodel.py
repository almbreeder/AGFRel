import sys
sys.path.append("../")

import config
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttentionLayer, EncoderLayer

class ObjectModel(nn.Module):
    
    def __init__(self, num_rels):
        
        super(ObjectModel, self).__init__()
        self.attention_heads = MultiHeadAttentionLayer(config.bert_feature_size, 12, 0.4)
        self.attention_tails = MultiHeadAttentionLayer(config.bert_feature_size, 12, 0.4)
        self.projection_heads = nn.Linear(config.bert_feature_size, num_rels).to(config.device)
        self.projection_tails = nn.Linear(config.bert_feature_size, num_rels).to(config.device)

        self.num_rels = num_rels

        self.rel_heads_attention = EncoderLayer(config.bert_feature_size,12,0.4,device='cuda')
        self.rel_tails_attention = EncoderLayer(config.bert_feature_size, 12, 0.4, device='cuda')
        self.rel_heads_linear = nn.Linear(config.bert_feature_size,num_rels).to(config.device)
        self.rel_tails_linear = nn.Linear(config.bert_feature_size,num_rels).to(config.device)
    
    def forward(self, bert_outputs, sub_head_batch, sub_tail_batch, attention_mask):
        
        sub_head_batch = sub_head_batch.unsqueeze(-1)
        sub_tail_batch = sub_tail_batch.unsqueeze(-1)
        
        batch_idx = torch.arange(0, list(sub_head_batch.size())[0]).unsqueeze(-1).to(config.device)
        
#         print(batch_idx.size())
#         print(sub_head_batch.size())
        
#         print(torch.cat([batch_idx, sub_head_batch], dim=1))

        sub_head_feature = gather_nd(bert_outputs, torch.cat([batch_idx, sub_head_batch], dim=1))
        sub_tail_feature = gather_nd(bert_outputs, torch.cat([batch_idx, sub_tail_batch], dim=1))
        sub_feature = torch.mean(torch.stack([sub_head_feature, sub_tail_feature], dim=1), dim=1)
        b, seq_len, feature_dim = list(bert_outputs.size())
        
        tokens_feature = sub_feature.unsqueeze(1).expand(b, seq_len, feature_dim).contiguous() + bert_outputs

        heads_outputs, _ = self.attention_heads(tokens_feature, tokens_feature, tokens_feature,attention_mask)
        tails_outputs, _ = self.attention_heads(tokens_feature, tokens_feature, tokens_feature,attention_mask)

        pred_obj_heads = self.projection_heads(heads_outputs)
        pred_obj_tails = self.projection_tails(tails_outputs)
        # [batch size, sentence len, relation nums]
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)

        rel_heads_output, _ = self.rel_heads_attention(tokens_feature + heads_outputs, attention_mask)
        rel_tails_output, _ = self.rel_tails_attention(tokens_feature + tails_outputs, attention_mask)
        pred_rel_heads = self.rel_heads_linear(rel_heads_output)
        pred_rel_tails = self.rel_tails_linear(rel_tails_output)

        return pred_obj_heads, pred_obj_tails, pred_rel_heads, pred_rel_tails

# paras: bert output [10,200,768]
def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    #[10,768]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1]:])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    return params[slices].view(*output_shape)
        
        