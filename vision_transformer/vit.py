
"""
Acknowledge: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """
    layer norm
    """
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class PatchEmbedStem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """

        Args:
            img_size:
            patch_size:
            in_chans:
            embed_dim:
        """
        super().__init__()
        self.img_height, self.img_width = pair(img_size)
        self.patch_height, self.patch_width = pair(patch_size)
        # num_patches = (self.img_width // patch_width) * (self.img_height // patch_height)
        # embed_dim = in_chans * self.patch_height * self.patch_width
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_height and W == self.img_width, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_height}*{self.img_width})."
        # x = self.proj(x).flatten(2).transpose(1, 2)  # (N, 14*14, 16*16X83) => (N, 196, 768)
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_head=8, dim_head=96, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """

        Args:
            dim:
            num_heads:
            dim_head:
            qkv_bias:
            qk_scale:
            attn_drop:
            proj_drop:
        """
        super(Attention, self).__init__()
        self.num_head = num_head
        inner_dim = num_head * dim_head
        proj_out = not (num_head == 1 and dim_head == dim)
        self.scale = qk_scale or dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)  # qkv weight
        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(proj_drop)
        ) if proj_out else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape  # (2, 197, 768)
        # (2, 197, 768) => (2, 197, 3 * 768) => (2, 197, 3, 8, 96) => (3, 2, 8, 197, 96)
        # qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # # q, k, v => (3, 2, 8, 197, 96)
        # q, k, v = qkv
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_head), qkv)

        # dots => (2, 8, 197, 197)
        # dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        dots = q @ k.transpose(-2, -1) * self.scale
        attn = self.attend(dots)
        attn = self.attn_drop(attn)
        # out => (2, 8, 197, 96)
        out = attn @ v
        # out => （2，197，96 * 8）
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out => (2, 197, 768)
        proj_out = self.proj_out(out)

        return proj_out


class FeedForward(nn.Module):
    """
    MLP
    """
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):

        return self.mlp(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, num_head, dim_head, dim_mlp, drop=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, num_head=num_head, dim_head=dim_head, proj_drop=drop)),
                PreNorm(dim, FeedForward(dim, hidden_dim=dim_mlp, drop=drop))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, num_head, dim_mlp, dim_head=96, pool='cls',
                 channels=3, drop=0., embed_drop=0.):
        """

        Args:
            image_size:
            patch_size:
            num_classes:
            dim: 16 * 16 * 3 = 768
            depth:
            num_head:
            dim_mlp:
            dim_head:
            pool:
            channels:
            drop:
            embed_drop:
        """
        super(ViT, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.embeding_stem = PatchEmbedStem(img_size=image_size, patch_size=patch_size, in_chans=channels, embed_dim=dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(embed_drop)

        self.transformer = Transformer(dim, depth, num_head=num_head, dim_head=dim_head, dim_mlp=dim_mlp, drop=drop)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.embeding_stem(x)
        b, n, _ = x.shape

        # concat cls token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # add position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer
        x = self.transformer(x)

        # cls head
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


def main():
    image_size = 224
    patch_size = 16
    num_patch = (224 // 16) * (224 // 16)
    embed_dim = patch_size * patch_size * 3
    image = torch.randn((2, 3, 224, 224))

    # test embedding
    # patch embedding
    patch_embed = PatchEmbedStem()
    embedding = patch_embed(image)
    print(embedding.shape)
    # position embedding
    pos_embed = nn.Parameter(torch.zeros(1, num_patch + 1, embed_dim))
    print(pos_embed.shape)
    # class token
    cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
    cls_tokens = cls_tokens.expand(2, -1, -1)

    x = torch.cat((cls_tokens, embedding), dim=1)
    x = x + pos_embed
    print(x.shape)

    # test attention
    attention = Attention(dim=768)
    proj_out = attention(x)
    print(proj_out.shape)

    # test FF
    mlp = FeedForward(dim=768, hidden_dim=768*4)
    out = mlp(x)
    print(out.shape)

    # test ViT
    vit = ViT(
        image_size=224,
        patch_size=32,
        num_classes=1000,
        dim=768,
        depth=6,
        num_head=16,
        dim_mlp=768*4,
        drop=0.1,
        embed_drop=0.1
    )

    vit.eval()
    img = torch.randn(1, 3, 224, 224)
    preds = vit(img)
    print(preds.shape)


if __name__ == "__main__":
    main()