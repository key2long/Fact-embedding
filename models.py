import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from functools import partial
from torch import einsum
from einops import rearrange
from utils import *
from seq2seq import Encoder, Decoder, Seq2Seq
import pdb


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)
    

class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


# class Denoise_Net(nn.Module):  
#     def __init__(self,
#                  input_dim=200,
#                  with_time_emb=True,
#                  max_seq_len=20,
#                  modelconfigs=None
#     ):
#         """

#         """
#         super().__init__()
  
#         assert modelconfigs is not None
#         self.model_name = modelconfigs['Denoise_Net']['name']
#         self.modelconfigs = modelconfigs['Denoise_Net']
#         # time embedding
#         dim = 64
#         time_dim = input_dim
#         if with_time_emb:
#             self.time_encoder = nn.Sequential(
#                 SinusoidalPositionEmbeddings(dim),
#                 nn.Linear(dim, time_dim),
#                 nn.GELU(),
#                 nn.Linear(time_dim, time_dim)
#             )
#         self.dense_fn = nn.Linear(input_dim*max_seq_len, input_dim)
#         if 'with_position_emb' in self.modelconfigs.keys():
#             # 这个是进行序列建模的
            
#             # position embedding
#             self.register_buffer("position_ids", torch.arange(
#                 max_seq_len).expand((1, -1)))
#             self.position_embeddings = nn.Embedding(
#                 max_seq_len, input_dim)

#             # self-attension
#             if self.model_name == 'transformer':
#                 te_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
#                                                       nhead=self.modelconfigs['nhead'])
#                 self.encoder = nn.TransformerEncoder(encoder_layer=te_layer,
#                                                      num_layers=self.modelconfigs['blocks_num'])
#             elif self.model_name == 'lstm':
#                 self.encoder = nn.LSTM(input_size=input_dim,
#                                        hidden_size=input_dim,
#                                        batch_first=False,
#                                        num_layers=2)

#             elif self.model_name == 'seq2seq':
#                 self.seq2seq_encoder = Encoder(input_size=input_dim,
#                                                hidden_size=256)
#                 self.seq2seq_decoder = Decoder(output_size=input_dim,
#                                                hidden_size=256)
#                 self.encoder = Seq2Seq(self.seq2seq_encoder,
#                                        self.seq2seq_decoder)
            
#         else:
#             # 这是整体的embed建模
#             self.input_drop = nn.Dropout(self.modelconfigs['input_drop'])
#             self.hidden_drop = nn.Dropout(self.modelconfigs['hidden_drop'])
#             self.feature_map_drop = nn.Dropout(self.modelconfigs['feat_drop'])
#             self.emb_dim1 = self.modelconfigs['embedding_dim1']
#             self.emb_dim2 = input_dim // self.emb_dim1
#             self.conv1 = nn.Conv2d(in_channels=5, 
#                                    out_channels=32,
#                                    kernel_size=(3, 3),
#                                    stride=1,
#                                    padding=0,
#                                    bias=self.modelconfigs['use_bias'])
#             self.bn0 = torch.nn.BatchNorm2d(5)
#             self.bn1 = torch.nn.BatchNorm2d(32)
#             self.dense_layer = nn.Linear(self.modelconfigs['hidden_size'], input_dim * 3)  # hidden_size 需要手动计算


#     def forward(self, 
#                 noisy_input_t, 
#                 condition,
#                 time):
#         '''
#         :param: noisy_input_t: b * path_len * dim at t moment or b * dim=>for conv
#         :param: condition: (h_emb, r_emb) or mask (h_emb, r_emb, ?)
#         :param: t: batch,
#         '''
#         seq_length = noisy_input_t.shape[1]
#         # pdb.set_trace()
#         c1_emb, c2_emb = condition
#         c1_emb, c2_emb = c1_emb.unsqueeze(1), c2_emb.unsqueeze(1) # b * 1 * dim
#         # pdb.set_trace()
#         condition_emb = torch.cat([c1_emb, c2_emb], dim=-1) # b * 1 * 2dim
#         condition_emb = condition_emb.expand(-1, seq_length, -1) # b * len * 2dim
#         # pdb.set_trace()
#         sequence = torch.cat((noisy_input_t, condition_emb), dim=-1) # b * len * 3dim
#         sequence = self.dense_fn(sequence) # b * len * 3dim => b * len * dim

#         # position embedding
#         if self.model_name == 'transformer':
#             position_ids = self.position_ids[:, :seq_length] # b * seq_length
#             time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1) # b * seq_length * dim

#             sequence = self.position_embeddings(position_ids) + sequence + time_embeddings
#             sequence = sequence.permute(1, 0, 2)  # [b, t, dim] => (t, b, dim)                        
#             sequence = self.encoder(sequence) # (t, b, dim)
#             sequence = sequence.permute(1, 0, 2)  # (t, b, dim) => [b, t, dim]   
#             return sequence 

#         elif self.model_name == 'seq2seq':
#             time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1) # b * seq_length * dim
#             sequence = sequence + time_embeddings # 不需要position  
#             sequence = sequence.permute(1, 0, 2) # (t, b, dim)
#             sequence = self.encoder(sequence) # (t, b, dim)
#             # (b, t, geometric_feature_dim + label_feature_dim)
#             sequence = sequence.permute(1, 0, 2)
#             return sequence 

#         elif self.model_name == 'lstm':
#             time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1) # b * seq_length * dim
#             sequence = sequence + time_embeddings # 不需要position  
#             sequence = sequence.permute(1, 0, 2) # (t, b, dim)
#             sequence = self.encoder(sequence) # (t, b, dim)
#             # (b, t, geometric_feature_dim + label_feature_dim)
#             sequence = sequence.permute(1, 0, 2)            
#             return sequence 

#         elif self.model_name == 'conv':
#             # 这个情况输入是 b * dim 没有序列长度的维度
#             noisy_input_t = noisy_input_t.view(-1, 3, self.emb_dim1, self.emb_dim2) # b * dim => batch * 3 * dim1 * dim2
#             c1_emb, c2_emb = condition
#             c1_emb = c1_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)  
#             c2_emb = c2_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
#             # pdb.set_trace()
#             stacked_inputs = torch.cat([noisy_input_t, c1_emb, c2_emb], dim=1) # batch * 3 + 2 * dim1 * dim2
#             stacked_inputs = self.bn0(stacked_inputs) # batch * 3+2 * 3dim1 * dim2
#             x = self.input_drop(stacked_inputs)
#             x = self.conv1(x) # batch * 32 * wdim * hdim
#             x = self.bn1(x) # batch * 32 * wdim * hdim
#             x = F.relu(x)
#             x = self.feature_map_drop(x)
#             x = x.view(x.shape[0], -1) # batch * hidden_dim => hidden_dim 需要计算 由wdim*hdim
#             x = self.dense_layer(x) # batch * embedding_dim 如果想要把 序列长度的维度 抽出来 reshap一下再接一个dense
#             x = x.view(x.shape[0], 3, -1)
#             return x


class Denoise_Net(nn.Module):  
    def __init__(self,
                 input_dim=200,
                 with_time_emb=True,
                 max_seq_len=20,
                 modelconfigs=None
    ):
        """

        """
        super().__init__()
  
        assert modelconfigs is not None
        self.model_name = modelconfigs['Denoise_Net']['name']
        self.modelconfigs = modelconfigs['Denoise_Net']
        # time embedding
        dim = 64
        time_dim = input_dim
        if with_time_emb:
            self.time_encoder = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        self.dense_fn = nn.Linear(input_dim, input_dim)
        if 'with_position_emb' in self.modelconfigs.keys():
            # 这个是进行序列建模的
            
            # position embedding
            self.register_buffer("position_ids", torch.arange(
                max_seq_len).expand((1, -1)))
            self.position_embeddings = nn.Embedding(
                max_seq_len, input_dim)

            # self-attension
            if self.model_name == 'transformer':
                te_layer = nn.TransformerEncoderLayer(d_model=input_dim, 
                                                      nhead=self.modelconfigs['nhead'])
                self.encoder = nn.TransformerEncoder(encoder_layer=te_layer,
                                                     num_layers=self.modelconfigs['blocks_num'])
            elif self.model_name == 'lstm':
                self.encoder = nn.LSTM(input_size=input_dim,
                                       hidden_size=input_dim,
                                       batch_first=False,
                                       num_layers=2)

            elif self.model_name == 'seq2seq':
                self.seq2seq_encoder = Encoder(input_size=input_dim,
                                               hidden_size=256)
                self.seq2seq_decoder = Decoder(output_size=input_dim,
                                               hidden_size=256)
                self.encoder = Seq2Seq(self.seq2seq_encoder,
                                       self.seq2seq_decoder)
            
        else:
            # 这是整体的embed建模
            self.input_drop = nn.Dropout(self.modelconfigs['input_drop'])
            self.hidden_drop = nn.Dropout(self.modelconfigs['hidden_drop'])
            self.feature_map_drop = nn.Dropout(self.modelconfigs['feat_drop'])
            self.emb_dim1 = self.modelconfigs['embedding_dim1']
            self.emb_dim2 = input_dim // self.emb_dim1
            self.conv1 = nn.Conv2d(in_channels=5, 
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding=0,
                                   bias=self.modelconfigs['use_bias'])
            self.bn0 = torch.nn.BatchNorm2d(5)
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.dense_layer = nn.Linear(self.modelconfigs['hidden_size'], input_dim * 3)  # hidden_size 需要手动计算


    def forward(self, 
                noisy_input_t, 
                condition,
                time):
        '''
        :param: noisy_input_t: b * path_len * dim at t moment or b * dim=>for conv
        :param: condition: (h_emb, r_emb) or mask (h_emb, r_emb, ?)
        :param: t: batch,
        '''
        seq_length = noisy_input_t.shape[1]
        # # pdb.set_trace()
        # c1_emb, c2_emb = condition
        # c1_emb, c2_emb = c1_emb.unsqueeze(1), c2_emb.unsqueeze(1) # b * 1 * dim
        # # pdb.set_trace()
        # condition_emb = torch.cat([c1_emb, c2_emb], dim=-1) # b * 1 * 2dim
        # condition_emb = condition_emb.expand(-1, seq_length, -1) # b * len * 2dim
        # # pdb.set_trace()
        # sequence = torch.cat((noisy_input_t, condition_emb), dim=-1) # b * len * 3dim
        # 这里输入的还是部分加噪的emb, 没有改变
        sequence = self.dense_fn(noisy_input_t) # b * len * 3dim => b * len * dim

        # position embedding
        if self.model_name == 'transformer':
            position_ids = self.position_ids[:, :seq_length] # b * seq_length
            time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1) # b * seq_length * dim

            sequence = self.position_embeddings(position_ids) + sequence + time_embeddings
            sequence = sequence.permute(1, 0, 2)  # [b, t, dim] => (t, b, dim)                        
            sequence = self.encoder(sequence) # (t, b, dim)
            sequence = sequence.permute(1, 0, 2)  # (t, b, dim) => [b, t, dim]   
            return sequence 

        elif self.model_name == 'seq2seq':
            time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1) # b * seq_length * dim
            sequence = sequence + time_embeddings # 不需要position  
            sequence = sequence.permute(1, 0, 2) # (t, b, dim)
            sequence = self.encoder(sequence) # (t, b, dim)
            # (b, t, geometric_feature_dim + label_feature_dim)
            sequence = sequence.permute(1, 0, 2)
            return sequence 

        elif self.model_name == 'lstm':
            time_embeddings = self.time_encoder(time).unsqueeze(1).expand(-1, seq_length, -1) # b * seq_length * dim
            sequence = sequence + time_embeddings # 不需要position  
            sequence = sequence.permute(1, 0, 2) # (t, b, dim)
            sequence = self.encoder(sequence) # (t, b, dim)
            # (b, t, geometric_feature_dim + label_feature_dim)
            sequence = sequence.permute(1, 0, 2)            
            return sequence 

        elif self.model_name == 'conv':
            # 这个情况输入是 b * dim 没有序列长度的维度
            noisy_input_t = noisy_input_t.view(-1, 3, self.emb_dim1, self.emb_dim2) # b * dim => batch * 3 * dim1 * dim2
            c1_emb, c2_emb = condition
            c1_emb = c1_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)  
            c2_emb = c2_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
            # pdb.set_trace()
            stacked_inputs = torch.cat([noisy_input_t, c1_emb, c2_emb], dim=1) # batch * 3 + 2 * dim1 * dim2
            stacked_inputs = self.bn0(stacked_inputs) # batch * 3+2 * 3dim1 * dim2
            x = self.input_drop(stacked_inputs)
            x = self.conv1(x) # batch * 32 * wdim * hdim
            x = self.bn1(x) # batch * 32 * wdim * hdim
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(x.shape[0], -1) # batch * hidden_dim => hidden_dim 需要计算 由wdim*hdim
            x = self.dense_layer(x) # batch * embedding_dim 如果想要把 序列长度的维度 抽出来 reshap一下再接一个dense
            x = x.view(x.shape[0], 3, -1)
            return x

class Triple_Encoder(nn.Module):
    def __init__(self,
                 entity_emb_dim=64,
                 max_seq_len=20,
                 modelconfigs=None
    ):
        super().__init__()

        assert modelconfigs is not None
        self.model_name = modelconfigs['Triple_Encoder']['name']
        self.modelconfigs = modelconfigs['Triple_Encoder']
        self.dense_fn = nn.Linear(
            entity_emb_dim, entity_emb_dim)
        # position embedding
        self.register_buffer("position_ids", torch.arange(
            max_seq_len).expand((1, -1)))
        self.position_embeddings = nn.Embedding(
            max_seq_len, entity_emb_dim)

        # self-attension
        if self.model_name == 'transformer':
            te_layer = nn.TransformerEncoderLayer(d_model=entity_emb_dim, 
                                                  nhead=self.modelconfigs['nhead'])
            self.encoder = nn.TransformerEncoder(encoder_layer=te_layer,
                                                 num_layers=self.modelconfigs['blocks_num'])
        elif self.model_name == 'lstm':
            self.encoder = nn.LSTM(input_size=entity_emb_dim,
                                   hidden_size=entity_emb_dim,
                                   batch_first=False,
                                   num_layers=2)

        elif self.model_name == 'seq2seq':
            self.seq2seq_encoder = Encoder(input_size=entity_emb_dim,
                                           hidden_size=256)
            self.seq2seq_decoder = Decoder(output_size=entity_emb_dim,
                                           hidden_size=256)
            self.encoder = Seq2Seq(self.seq2seq_encoder, self.seq2seq_decoder)
        
        elif self.model_name == 'conv':
            self.input_drop = nn.Dropout(self.modelconfigs['input_drop'])
            self.hidden_drop = nn.Dropout(self.modelconfigs['hidden_drop'])
            self.feature_map_drop = nn.Dropout(self.modelconfigs['feat_drop'])
            self.emb_dim1 = self.modelconfigs['embedding_dim1']
            self.emb_dim2 = entity_emb_dim // self.emb_dim1
            self.conv1 = nn.Conv2d(in_channels=3, 
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding=0,
                                   bias=self.modelconfigs['use_bias'])
            self.bn0 = torch.nn.BatchNorm2d(3)
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.dense_layer = nn.Linear(self.modelconfigs['hidden_size'], entity_emb_dim * 3)  # hidden_size 需要手动计算
                      

    def forward(self, h_emb, path_emb, t_emb):
        '''
        param: h_emb, t_emb: b * dim
        param: path_emb: b * dim or b * path_len * dim
        '''
        # 如果是用序列建模的方式编码
        if self.model_name != 'Conv':
            if len(path_emb.shape) == 3: # 这时path不止一个关系
                h_emb = h_emb.unsqueeze(1) # b * 1 * dim
                t_emb = t_emb.unsqueeze(1)
                sequence = torch.cat((h_emb, path_emb, t_emb), dim=1) # b * path_len+2 * dim
            else:
                sequence = torch.stack((h_emb, path_emb, t_emb), dim=1) # b * dim, b * dim, b * dim => b * 3 * dim
            
            seq_length = sequence.size(1)


            # [b, t, dim]
            sequence = self.dense_fn(sequence)

            if self.model_name == "transformer":
                position_ids = self.position_ids[:, :seq_length] # b * seq_length
                sequence = self.position_embeddings(position_ids) + sequence
                sequence = sequence.permute(1, 0, 2)  # [b, t, dim] => (t, b, dim)                        
                sequence = self.encoder(sequence)
                sequence = sequence.permute(1, 0, 2)  # (t, b, dim) => [b, t, dim]  

            if self.model_name == "lstm":
                sequence = sequence.permute(1, 0, 2)
                sequence = self.encoder(sequence)  
                sequence = sequence[0].permute(1, 0, 2) # (t, b, dim) => [b, t, dim]          

            if self.model_name == "seq2seq":
                sequence = sequence.permute(1, 0, 2)
                sequence = self.encoder(sequence) 
                sequence = sequence.permute(1, 0, 2) # (t, b, dim) => [b, t, dim]
            return sequence   
        
        else: # 如果是用ConvNet编码的情况
            h_emb = h_emb.view(-1, 1, self.emb_dim1, self.emb_dim2) # batch * 1 * dim1 * dim2
            path_emb = path_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)  
            t_emb = t_emb.view(-1, 1, self.emb_dim1, self.emb_dim2)
            stacked_inputs = torch.cat([h_emb, path_emb, t_emb], dim=1) # batch * 3 * dim1 * dim2
            stacked_inputs = self.bn0(stacked_inputs) # batch * 1 * 3dim1 * dim2
            x = self.input_drop(stacked_inputs)
            x = self.conv1(x) # batch * 32 * wdim * hdim
            x = self.bn1(x) # batch * 32 * wdim * hdim
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(x.shape[0], -1) # batch * hidden_dim
            x = self.dense_layer(x) # batch * embedding_dim 如果想要把维度抽出来 reshap一下再接一个dense
            x = x.view(x.shape[0], 3, -1)
            return x


# class Triple_Classifier(nn.Module):
#     def __init__(self, n_layers, input_dim, nentity):
#         # 这里的input_dim需要用 seq_len * dim
#         super().__init__()
#         self.n_layers = n_layers
#         self.input_dim = input_dim
#         self.input_dim10 = input_dim * 10 
#         self.nentity = nentity

#         setattr(self, f"classifier_layer_{int(1)}", nn.Linear(self.input_dim, self.input_dim10))
#         for i in range(2, self.n_layers + 1):
#             setattr(self, f"classifier_layer_{i}", nn.Linear(self.input_dim10, self.input_dim10))

#         self.last_layer = nn.Linear(self.input_dim10, nentity)
    
#     def forward(self, x):
#         batch = x.shape[0]
#         x = x.reshape(batch, -1)
#         for i in range(1, self.n_layers + 1):
#             x = F.relu(getattr(self, f"classifier_layer_{i}")(x))
#         x = self.last_layer(x)
#         # output = F.relu(x)
#         output = x
#         return output   

# class Triple_Classifier(nn.Module):
#     def __init__(self, n_layers, input_dim, nentity):
#         # 这里的input_dim需要用 seq_len + 2 * dim
#         super().__init__()
#         self.n_layers = n_layers
#         self.input_dim = input_dim
#         self.nentity = nentity
#         for i in range(1, self.n_layers + 1):
#             setattr(self, f"classifier_layer_{i}", nn.Linear(input_dim, input_dim))
#         self.last_layer = nn.Linear(input_dim, nentity)
    
#     def forward(self, x, condition):
#         batch = x.shape[0]
#         cond1, cond2 = condition
#         condition = torch.cat((cond1.unsqueeze(1), cond2.unsqueeze(1)), dim=1) # b * 2 * dim
#         input = torch.cat((x, condition), dim=1) # b * seq_len + 2, dim
#         input = input.reshape(batch, -1) # b * [(seq_len + 2) * dim]
#         for i in range(1, self.n_layers + 1):
#             input = F.relu(getattr(self, f"classifier_layer_{i}")(input))
#         input = self.last_layer(input)
#         # output = F.relu(x)
#         output = input
#         return output  


def triple_classifier(triple_embedding, entity_embedding, mode):
    # triple_embedding: pred embedding :b * 3 * dim
    # 

    head_embedding = triple_embedding[:, 0, :]
    relation_embedding = triple_embedding[:, 1, :]
    tail_embedding = triple_embedding[:, 2, :]
    emb_norm = (entity_embedding ** 2).sum(-1).view(-1, 1)  # vocab, 1

    # text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
    # arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
    # dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(entity_embedding,
    #                                                             text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
    # scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
    #                                                     hidden_repr.size(1)) # vocab, bsz*seqlen
    # scores = -scores.permute(1, 2, 0).contiguous()


    if mode == 'head-batch':
        text_emb = head_embedding # bsz, dim
        text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz 
        arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz, 1
        dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(entity_embedding,
                                                                    text_emb_t)  # (vocab, d) x (d, bsz)
        scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), head_embedding.size(0)) # vocab, bsz
        scores = scores.permute(1, 0).contiguous()
        # pdb.set_trace()

        # entity_emb_norm = (entity_embedding**2).sum(-1).view(-1, 1) # vocab * dim => vocab * 1

        # score = torch.mm(head_embedding, entity_embedding.T)
    
    if mode == 'tail-batch':
        text_emb = tail_embedding # bsz, dim
        text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz 
        arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz, 1
        dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(entity_embedding,
                                                                    text_emb_t)  # (vocab, d) x (d, bsz)
        scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), head_embedding.size(0)) # vocab, bsz
        scores = scores.permute(1, 0).contiguous()
        # pdb.set_trace()

        # text_emb = tail_embedding
        # score = torch.mm(tail_embedding, entity_embedding.T)
    
    return scores



class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


if  __name__ == '__main__':
    pass