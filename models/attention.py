import torch
import torch.nn as nn
import config


class FilterModel(nn.Module):
    def __init__(self,rel_num):

        self.layer1 = EncoderLayer(config.bert_feature_size,12,0.4,device='cuda')
        self.layer2 = EncoderLayer(config.bert_feature_size, 12, 0.4, device='cuda')
        self.layer3 = EncoderLayer(config.bert_feature_size, 12, 0.4, device='cuda')
        self.start_linear = nn.Linear(config.bert_feature_size,rel_num+1)
        self.end_linear = nn.Linear(config.bert_feature_size, rel_num + 1)

    def forward(self,layer1_input,layer2_input,layer3_input, src_mask):
        hidden, obj_auxiliary = layer1_input
        batch_size, src_len, hid_dim = hidden.shape

        src = torch.cat([hidden, obj_auxiliary], dim=2)

        src1, att1 = self.atten_layer1(src, src_mask)
        src2 = torch.cat([src1, layer2_input], dim=2)
        src2, att2 = self.atten_layer2(src2, src_mask)
        src3 = torch.cat([src2, layer3_input], dim=2)
        src3, att3 = self.atten_layer3(src3, src_mask)

        relation_head_logists = self.start_linear(src2)
        relation_tail_logists = self.end_linear(src3)

        return relation_head_logists, relation_tail_logists




class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     hid_dim*2,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        # self attention (Q,K,V)
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        # src = [batch size, src len, hid dim]
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        att = src

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # # dropout, residual connection and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src, att

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device='cuda'):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        # Q,V,K = [hid dim, hid dim]
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query = [batch size, src len, hid dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # attentionMatrix = [batch size, n heads, src len, src len]
        attentionMatrix = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        mask = mask.unsqueeze(1).unsqueeze(2)

        if mask is not None:
            attentionMatrix = attentionMatrix.masked_fill(mask == 0, -1e10)

        # dim=0 对每一行进行softmax, dim=-1 对每一列进行softmax
        # attentionMatrix = F.softmax(attentionMatrix, dim=-1)

        attentionMatrix = torch.softmax(attentionMatrix, dim=-1)

        # kk_att = attentionMatrix.squeeze(0).data.cpu().numpy()

        # x = [batch size, n heads, src len, head dim]
        x = torch.matmul(self.dropout(attentionMatrix), V)

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, src len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attentionMatrix



class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x