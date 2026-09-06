import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, PNAConv
from mamba_ssm import Mamba

import math
import numpy as np
from functools import partial

from .soft_moe import SoftMoELayerWrapper
from .lm_mamba import BiDirectionMixerModel

from timm.models.layers import drop_path
from torch_geometric.utils import degree, to_dense_batch

class PNA(nn.Module):
    def __init__(self, input_dim=15, out_put_dim=128):
        super(PNA, self).__init__()
        self.conv1 = PNAConv(input_dim, out_put_dim, deg=torch.tensor(30.),
                             aggregators=['mean', 'min', 'max'],
                             scalers=['identity', 'linear'])
        self.conv2 = PNAConv(out_put_dim, out_put_dim, deg=torch.tensor(30.),
                             aggregators=['mean', 'min', 'max'],
                             scalers=['identity', 'linear'])
        self.norm1 = nn.LayerNorm(out_put_dim)
        self.norm2 = nn.LayerNorm(out_put_dim)

    def forward(self, data):  
        x, edge_index = data.x, data.edge_index  
        x = self.conv1(x, edge_index)              
        x = F.gelu(x)   
        x = self.norm1(x)                            
        x = self.conv2(x, edge_index)
        x = F.gelu(x) 
        x = self.norm2(x)           
        return x

class GAT(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=256, out_put_dim=128, heads=4, dropout=0.1):
        super(GAT, self).__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.conv1 = GATConv(
            input_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            hidden_dim,
            out_put_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_put_dim)

    def forward(self, data):  
        x, edge_index = data.x, data.edge_index  
        x = self.conv1(x, edge_index)              
        x = F.gelu(x)   
        x = self.norm1(x)                            
        x = self.conv2(x, edge_index)
        x = F.gelu(x) 
        x = self.norm2(x)                     
        return x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    
class Conv1d(nn.Module):
    default_act = nn.GELU()
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,norm=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.norm = nn.LayerNorm(c2) if norm is True else nn.Identity()      
        # self.norm = nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x).permute(0,2,1)
        return self.act(self.norm(x))
    
class Conv1d_Pool(nn.Module):
    default_act = nn.GELU()
    
    def __init__(self, c1, c2, out_size, k=1, s=1, p=None, g=1, d=1, act=True,norm=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(out_size)
        self.norm = nn.LayerNorm(c2) if norm is True else nn.Identity()      
        # self.norm = nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool(self.conv(x)).permute(0,2,1)
        return self.act(self.norm(x))

class Embedding(nn.Module):
    def __init__(self, Bilstm_input_feature_size, vocab_size, max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, Bilstm_input_feature_size)
        self.pos_embed = nn.Embedding(max_len, Bilstm_input_feature_size)
        self.norm = nn.LayerNorm(Bilstm_input_feature_size)
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding

class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self, Bilstm_output_feature_size, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = torch.nn.Linear(Bilstm_output_feature_size*2, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, Bilstm_output_feature_size*2)
        self.relu = torch.nn.GELU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class Muti_kernal_conv_mamba(nn.Module):
    def __init__(self, embed_dim=128, output_dim=128, num_bi_lstm_layer=4, out_conv_dim=64, num_conv=6, max_len=31, vocab_size=21):
        super(Muti_kernal_conv_mamba,self).__init__()
        self.embedding = Embedding(embed_dim, vocab_size, max_len)
        self.bi_lstm = nn.LSTM(embed_dim, output_dim, num_bi_lstm_layer, batch_first=True, bidirectional=True)

        self.feature_learn = nn.Sequential(
            torch.nn.Linear(num_conv*out_conv_dim, embed_dim-1),
            torch.nn.GELU(),)
        
        self.conv_pool_list = nn.ModuleList()
        self.mamba_list = nn.ModuleList()
        for i in range(num_conv):
            self.conv_pool_list.append(Conv1d_Pool(embed_dim*2, out_conv_dim, max_len, i+2))
            ssm_cfg = {'d_conv': 2}
            self.mamba_list.append(BiDirectionMixerModel(d_model=out_conv_dim, n_layer=2, ssm_cfg=ssm_cfg, rms_norm=True, fused_add_norm=True))

    def forward(self, input_ids, device):
        input_ids = input_ids.to(device)
        output_embedding = self.embedding(input_ids)
        output, _ = self.bi_lstm(output_embedding)
        out_all = []
        for (conv_pool, mamba_block) in zip(self.conv_pool_list, self.mamba_list):
            x = conv_pool(output)
            x = mamba_block(x)
            out_all.append(x)
        out_all = torch.cat(out_all, dim=-1)
        seq_feature = self.feature_learn(out_all)
        return seq_feature

# ----------------------------------------------

class Attention(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads, proj_drop=0.):
        super().__init__()
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads,batch_first=True)
        self.identity = nn.Identity()
        
        self.proj = nn.Linear(c, c)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        (out, attn_map) = self.ma(x,x,x)
        attn_map = self.identity(attn_map)
        
        x = self.proj(out)
        x = self.proj_drop(x)
        
        return x
    
class Cross_Attention(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads, proj_drop=0.):
        super().__init__()
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True)
        self.identity = nn.Identity()
        
        self.proj = nn.Linear(c, c)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x_query, x_key, key_padding_mask=None):
        (out, attn_map) = self.ma(x_query, x_key, x_key, key_padding_mask=key_padding_mask)
        attn_map = self.identity(attn_map)
        
        x = self.proj(out)
        x = self.proj_drop(x)
        
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, num_experts=16):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # TODO: MoE
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.num_experts = num_experts
        self.slots_per_expert = 1
        self.mlp = partial(
            SoftMoELayerWrapper,
            layer=Mlp,
            dim=dim,
            num_experts=self.num_experts,
            slots_per_expert=self.slots_per_expert,
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)()

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    
class cross_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, num_experts=16):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.attn = Cross_Attention(dim, num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # TODO: MoE
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.num_experts = num_experts
        self.slots_per_expert = 1
        self.mlp = partial(
            SoftMoELayerWrapper,
            layer=Mlp,
            dim=dim,
            num_experts=self.num_experts,
            slots_per_expert=self.slots_per_expert,
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)()

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x_query, x_key, key_padding_mask=None):
        if self.gamma_1 is None:
            x = x_query + self.drop_path(self.attn(x_query, x_key, key_padding_mask=key_padding_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x_query + self.drop_path(self.gamma_1 * self.attn(x_query, x_key, key_padding_mask=key_padding_mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    
class Structure_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure_feature_learn_complete_block = nn.Sequential(
            nn.Linear(34, 34//2),
            nn.GELU(),
            nn.Linear(34//2, 34))
        self.structure_feature_learn_AA_block = nn.Sequential(
            nn.Linear(43, 43//2),
            nn.GELU(),
            nn.Linear(43//2, 43),)
        self.structure_feature_learn_atom_block = nn.Sequential(
            nn.Linear(35, 35//2),
            nn.GELU(),
            nn.Linear(35//2, 35),)
        self.feature_learned = nn.Sequential(
            nn.Linear(112, 112 // 2),
            nn.GELU(),
            nn.Linear(112 // 2, 112//4),)
        # self.feature_final = nn.Sequential(
        #     nn.Linear(112//4, 2),)
        # self.combine_block_MLP = nn.Sequential(
        #     nn.Linear(2, 1),
        #     nn.Sigmoid())
        
    def forward(self, structure_feature):
        x_Complete = self.structure_feature_learn_complete_block(structure_feature[:, 0:(13 + 21)])
        x_AA = self.structure_feature_learn_AA_block(structure_feature[:, (13 + 21):(34 + 34 + 9)])
        x_ATOM = self.structure_feature_learn_atom_block(structure_feature[:, (34 + 34 + 9):])
        x1 = torch.cat((x_Complete, x_AA, x_ATOM), dim=1)
        x= self.feature_learned(x1)
        return x

class PTM_MoE(nn.Module):
    def __init__(self, seq_embed_dim, embed_dim, num_heads, seq_window=15, moe_num_experts=16):
        super().__init__()
        # self.structure_layer = Structure_layer()
        self.seq_window = seq_window
        self.moe_num_experts = moe_num_experts
        self.structure_layer = GAT(hidden_dim=embed_dim * 2, out_put_dim=embed_dim)
        self.seq_embed=Muti_kernal_conv_mamba(embed_dim=seq_embed_dim, output_dim=seq_embed_dim)
        self.motif_mlp = Mlp(in_features=2*seq_window+1,out_features=31)
        # self.fusion_mlp = Mlp(in_features=95,hidden_features=embed_dim,out_features=embed_dim)

        # self.attn_block = nn.Sequential(*[cross_block(dim=embed_dim, num_heads=num_heads) for _ in range(1)])
        self.attn_block = cross_block(dim=embed_dim, num_heads=num_heads, num_experts=moe_num_experts)
        self.seq_norm = nn.LayerNorm(seq_embed_dim)
        self.seq_proj = nn.Linear(seq_embed_dim, embed_dim) if seq_embed_dim != embed_dim else nn.Identity()
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.classifier = torch.nn.Sequential(nn.Linear(embed_dim, 1),)
        # self.focal_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.initialize_weights()

    def initialize_weights(self):
        # torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m, fix_group_fanout=True):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            if fix_group_fanout:
                fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _num_graphs(self, graph_data):
        if hasattr(graph_data, "num_graphs"):
            return graph_data.num_graphs
        return 1

    def _reshape_graph_attr(self, value, num_graphs):
        if value.dim() == 1:
            return value.view(num_graphs, -1)
        if value.size(0) != num_graphs:
            return value.view(num_graphs, -1)
        return value

    def forward(self, graph_data):
        # modif_site = graph_data.modif_site
        num_graphs = self._num_graphs(graph_data)
        x_motif = self._reshape_graph_attr(graph_data.sequence_feature, num_graphs)
        x_seq = self._reshape_graph_attr(graph_data.aa_id, num_graphs)

        x_struct = self.structure_layer(graph_data)
        if hasattr(graph_data, "batch") and graph_data.batch is not None:
            x_struct, struct_mask = to_dense_batch(x_struct, graph_data.batch)
            struct_padding_mask = ~struct_mask
        else:
            x_struct = x_struct.unsqueeze(0)
            struct_padding_mask = None

        x_motif = self.motif_mlp(x_motif)
        x_seq = self.seq_embed(x_seq, x_motif.device)
        x_seq = torch.cat([x_seq, x_motif.unsqueeze(-1)], dim=-1)
        x_seq = self.seq_norm(x_seq)
        x_seq = self.seq_proj(x_seq)

        # x_motif_struct = torch.cat([x_struct, x_motif], dim=1)
        # x_motif_struct = self.fusion_mlp(x_motif_struct).unsqueeze(1)

        # x_all_fusion = torch.cat([x_motif_struct, x_seq], dim=1)
        # x_all_fusion = self.norm(x_all_fusion)

        # cross attention
        x_all_fusion = self.attn_block(x_seq, x_struct, key_padding_mask=struct_padding_mask)
        x_all_fusion = x_all_fusion.mean(dim=1)
        x_all_fusion = self.fc_norm(x_all_fusion)
        
        logit= self.classifier(x_all_fusion.clone())

        return logit


if __name__ == '__main__':
    model = PTM_MoE(128,128,4)
    print(model)
