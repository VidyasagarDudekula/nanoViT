import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelArgs
import math

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')



class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.seq_len = config.max_seq_len
        pos_emb = torch.ones(self.seq_len, self.dim)
        t = torch.arange(0, self.seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000)/self.dim))
        cos = torch.cos(t*div)
        sin = torch.sin(t*div)
        pos_emb[:, ::2] = sin
        pos_emb[:, 1::2] = cos
        self.register_buffer('pos_embbed', pos_emb)
    
    def forward(self, x):
        B, T, C = x.shape
        x += self.pos_embbed[:T, :].requires_grad_(False)
        return x

def preprocess_thta_values(head_dim, seq_len, base):
    theta = 1.0/ (base ** (torch.arange(0, head_dim, 2)/head_dim))

    t = torch.arange(0, seq_len)
    values = torch.einsum("i,j->ij", t, theta)
    freqs = torch.cat((values, values), dim=-1)
    return freqs


def rotate_shift(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPEmbedding(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.head_dim = config.head_dim
        self.base = config.base
        self.seq_len = config.max_seq_len
        self.freqs = preprocess_thta_values(self.head_dim, self.seq_len, self.base)
        self.register_buffer('cos', torch.cos(self.freqs))
        self.register_buffer('sin', torch.sin(self.freqs))
    
    def forward(self, x):
        # B, n_heads, T, head_dim
        B, H, T, C = x.shape
        cos = self.cos[:T, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T, :].unsqueeze(0).unsqueeze(0)
        altered_x = rotate_shift(x)
        return (x*cos + altered_x*sin)

class PatchEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.patch_size = config.patch_size
        self.seq_len = config.max_seq_len
        self.conv1 = nn.Conv2d(config.in_channels, self.dim, kernel_size=self.patch_size, stride=self.patch_size, bias=True)
    
    def forward(self, x: torch.Tensor):
        B, IN_CH, H, W = x.shape
        x = self.conv1(x)
        x = x.flatten(2) # B, dim, patch_size*patch_size
        x = x.transpose(-2, -1) # B, Patch_size * Patch_size, dim 
        return x


class LayerNormalizations(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.eps = config.eps
        self.alpha = nn.Parameter(torch.ones(config.dim))
        self.beta = nn.Parameter(torch.zeros(config.dim))
    
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        value = (x-mean)/((var + self.eps)**0.5)
        return value * self.alpha + self.beta

class ResidualNetwrorks(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        pass
    
    def forward(self, x, prev_layer_ouput):
        x = x + prev_layer_ouput
        return x



class SelfAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.n_q_head = config.n_q_head
        self.n_kv_head = config.n_kv_head
        self.n_kv_group = config.n_kv_group
        self.head_dim = config.head_dim
        self.wq = nn.Linear(self.dim, self.head_dim * self.n_q_head)
        self.wk = nn.Linear(self.dim, self.head_dim * self.n_kv_head)
        self.wv = nn.Linear(self.dim, self.head_dim*self.n_kv_head)
        self.wo = nn.Linear(self.dim, self.dim)
        self.attns_dp = nn.Dropout(config.dropout)
        self.proj_dp = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_q_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(1, 2).repeat_interleave(self.n_kv_group, dim=1)
        v = self.wv(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(1, 2).repeat_interleave(self.n_kv_group, dim=1)

        qk = q @ k.mT
        attns = qk * self.head_dim** -0.5
        attns = F.softmax(attns, dim=-1)
        attns = attns @ v
        attns = attns.transpose(1, 2).reshape(B, T, C)
        attns = self.attns_dp(attns)
        attns = self.wo(attns)
        attns = self.proj_dp(attns)
        return attns

class SwiGLU(nn.Module):
    def forward(self, x):
        return x * F.silu(x)


class FeedForwardLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = int(((config.dim * 2)/3) * 4)
        
        self.w1 = nn.Linear(self.dim, self.hidden_dim)
        self.w2 = nn.Linear(self.dim, self.hidden_dim)
        self.w3 = nn.Linear(self.hidden_dim, self.dim)
        self.dp = nn.Dropout(config.dropout)
        self.activation = SwiGLU()
    
    def forward(self, x: torch.Tensor):
        gate = self.activation(self.w1(x))
        value = self.w2(x)
        x = self.w3(gate * value)
        x = self.dp(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.res = ResidualNetwrorks(config)
        self.norm1 = LayerNormalizations(config)
        self.norm2 = LayerNormalizations(config)
        self.self_attn = SelfAttention(config)
        self.ffn = FeedForwardLayer(config)
    
    def forward(self, x):
        x = self.res(x, self.self_attn(self.norm1(x)))
        x = self.res(x, self.ffn(self.norm2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_layers = config.n_encoder
        self.layers = nn.Sequential(*[EncoderBlock(config) for _ in range(self.n_layers)])
    
    def forward(self, x):
        x = self.layers(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.patch_emb = PatchEmbeddings(config)
        self.dim = config.dim
        self.pos_emb = PositionalEmbedding(config)
        self.num_classes = config.num_classes
        self.seq_len = config.max_seq_len
        self.proj_layer = nn.Linear(config.dim, self.num_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        torch.nn.init.xavier_uniform_(self.cls_token)
        self.encoder = Encoder(config)
    
    def forward(self, x):
        x = self.patch_emb(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=-2)
        x = self.pos_emb(x)
        x = self.encoder(x)
        x = x[:, 0, :]
        x = self.proj_layer(x)
        return x
        

if __name__ == '__main__':
    config = ModelArgs()
    x = torch.randn((10, config.in_channels, config.image_size, config.image_size))
    m = VisionTransformer(config)
    x = m(x)
    print(x.shape)
        



