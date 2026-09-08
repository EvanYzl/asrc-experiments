import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import einops
from copy import deepcopy


rspmm = load(
    name="rspmm",
    sources=[
        "source/rspmm_ops.cpp",
        "source/rspmm_ops.cu",
    ],
    extra_cuda_cflags=["-O3"],
    keep_intermediates=False
)


class RSPMMAddMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, edge_type, relation, x):
        output = rspmm.mul_sum_forward(edge_index, edge_type, relation, x)
        ctx.save_for_backward(edge_index, edge_type, relation, x, output)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        relation_grad, input_grad = rspmm.mul_sum_backward(*ctx.saved_tensors, grad_outputs[0])
        return None, None, relation_grad, input_grad


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias


NORM_CLS = {
    "LN": nn.LayerNorm,
    "TANH": DynamicTanh
}


class FeedforwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = nn.LeakyReLU()
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class RelationalMessagePassingLayer(nn.Module):
    def __init__(self, num_relations: int, hidden_size: int, intermediate_size: int, dropout: float, norm: str):
        super().__init__()
        self.z_proj = nn.Linear(hidden_size, num_relations * hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.norm = NORM_CLS[norm](hidden_size)
        self.ffn = FeedforwardNetwork(hidden_size, intermediate_size, dropout)

    def forward(self, edge_index, edge_type, z, x):
        hidden_size = x.size(-1)
        nodes = einops.rearrange(x, 'b e d -> e (b d)')
        edges = einops.rearrange(self.z_proj(z), 'b (r d) -> r (b d)', d=hidden_size)
        out = RSPMMAddMulFunction.apply(edge_index, edge_type, edges, nodes)
        out = einops.rearrange(out, 'e (b d) -> b e d', d=hidden_size)

        out = out * torch.sigmoid(self.gate(x))
        out = self.ffn(out)
        out = self.norm(out)
        out = out + x
        return out


class RelationalGraphNetwork(nn.Module):
    def __init__(self, num_relations: int, num_layers: int, hidden_size: int, intermediate_size: int, dropout: float, norm: str):
        super().__init__()
        layer = RelationalMessagePassingLayer(num_relations, hidden_size, intermediate_size, dropout, norm)
        self.layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])

    def forward(self, edge_index, edge_type, z, x):
        for layer in self.layers:
            x = layer(edge_index, edge_type, z, x)
        return x


class GatedAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, intermediate_size: int, num_leaders: int, norm: str):
        super().__init__()
        self.num_heads, self.dropout = num_heads, dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.g_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.input_layer_norm = NORM_CLS[norm](hidden_size)
        self.output_layer_norm = NORM_CLS[norm](hidden_size)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.ReLU()

        self.num_leaders = num_leaders
        self.f_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_size, num_leaders))
        self.b_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_size, num_leaders))

    def forward(self, x):
        bz, m, n = x.shape[0], x.shape[1], self.num_leaders

        w0 = self.f_proj(x)  # b, m, n
        hidden_states = torch.einsum('bmd,bmn->bnd', x, torch.softmax(w0, dim=1))

        residual = hidden_states

        hidden_states = self.input_layer_norm(hidden_states)

        query_states = self.q_proj(hidden_states).view(bz, n, self.num_heads, -1).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape(bz, n, self.num_heads, -1).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(bz, n, self.num_heads, -1).transpose(1, 2)
        gate_states = self.g_proj(hidden_states).reshape(bz, n, self.num_heads, -1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (N, H, seq, D)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output * torch.sigmoid(gate_states)
        attn_output = attn_output.reshape(bz, n, -1)

        hidden_states = self.o_proj(attn_output) + residual
        residual = hidden_states
        hidden_states = self.output_layer_norm(hidden_states)
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

        out = hidden_states + residual

        w1 = self.b_proj(x)
        out = torch.einsum('bnd,bmn->bmd', out, torch.softmax(w1, dim=-1))
        return out
