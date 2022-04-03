import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=2, img_dim=32, patch_dim=4, embed_dim=216):
        super(PatchEmbed, self).__init__()
        self.n_patches = (img_dim // patch_dim) ** 2
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_dim, patch_dim), stride=(patch_dim, patch_dim))

    def forward(self, x):
        # X SHAPE IS (batch, in_channels, img_dim,  img_dim )
        x = self.conv(x)
        # new shape of x is going to be (batch, embed_dim , self.n_patches, self.n_patches)
        x = x.reshape(x.shape[0], self.embed_dim, -1)
        x = x.transpose(2, 1)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim=216, n_heads=6, qkv_bias=False, atttn_p=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = (embed_dim//n_heads)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.drop = nn.Dropout2d(atttn_p)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, key, query, value):
        # key ,query, value shape is [batch, n_pathes +1, embed_dim]
        key = self.k_linear(key).reshape(key.shape[0], self.n_heads, key.shape[1], self.head_dim)
        query = self.q_linear(query).reshape(query.shape[0], self.n_heads, query.shape[1], self.head_dim)
        value = self.v_linear(value).reshape(value.shape[0], self.n_heads, value.shape[1], self.head_dim)
        attn_score = torch.einsum("bnqd, bnkd -> bnqk", query, key)
        attn_soft = self.soft(attn_score)
        out = torch.einsum("bnqk, bnvd -> bnqd", attn_soft, value)
        out = self.drop(out)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return out


class MLP(nn.Module):
    def __init__(self, in_features=216, factor=2):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_features, factor * in_features)
        self.act = nn.GELU()
        self.l2 = nn.Linear(factor * in_features, in_features)

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        return x


class Block(nn.Module):
    def __init__(self, embed_dim=216, n_heads=6, qkv_bias=False, attn_p=0.0, p=0.0, factor=2):
        super(Block, self).__init__()
        self.attention = Attention(embed_dim, n_heads, qkv_bias,  attn_p)
        self.mlp = MLP(embed_dim, factor)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = x + self.norm1(self.attention(x, x, x))
        out = x + self.norm2(self.mlp(out))
        return out


class VisionTransformer(nn.Module):
    def __init__(self, img_dim=32, in_channels=2, patch_dim=4, embed_dim=216, n_heads=6, attn_p=0.,
                 n_classes=2, qkv_bias=False, n_blocks=2):
        super(VisionTransformer, self).__init__()
        self.block = Block(
            embed_dim, n_heads, qkv_bias, attn_p, p=0., factor=2
        )
        self.layers = nn.ModuleList()
        self.n_layers = n_blocks
        self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.patch = PatchEmbed(in_channels, img_dim, patch_dim, embed_dim=embed_dim)
        self.position_embedding = nn.Parameter(
            torch.ones(1, 1+self.patch.n_patches, embed_dim)
        )
        self.final = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding

        for i in range(self.n_layers):
            x = self.block(x)

        x = x[:, 0]
        out = self.final(x)
        return out