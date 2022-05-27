from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import cv2

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(self, img_size=640, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
    
        self.embeding = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.embeding(x).flatten(2).transpose(1, 2)
        return x  # x.shape is [8, 196, 512]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temp = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v):
        attention = torch.matmul(q/self.temp, k)

        # if mask is not None:
        #     attention = attention.masked_fill(mask==0, -1e9)
        attention = self.dropout(torch.softmax(attention, -1))
        out = torch.matmul(attention, v)
        return out

class Multihead(nn.Module):
    def __init__(self, d_feature, d_k, d_v, head, dropout=0.1):
        super().__init__()

        self.d_feature = d_feature
        self.d_k = d_k   # d_k = d_q
        self.d_v = d_v
        self.head = head

        self.fc_q = nn.Linear(d_feature, self.head * self.d_k)
        self.fc_k = nn.Linear(d_feature, self.head * self.d_k)
        self.fc_v = nn.Linear(d_feature, self.head * self.d_v)
        self.fc_o = nn.Linear(self.head * self.d_v, d_feature)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_feature, eps=1e-6)

        self.attn = ScaledDotProductAttention(temperature=d_k**0.5)

    def forward(self, q, k, v):
        """
        q: (b, len_q, d_feature)
        q=k=v
        """
        batch, len_q = q.shape[:2]
        len_k, len_v = k.shape[1], v.shape[1]

        shortcut_v = v

        q = self.fc_q(q).view(batch, len_q, self.head, self.d_k).permute(0,2,1,3)
        k = self.fc_k(k).view(batch, len_k, self.head, self.d_k).permute(0,2,3,1)
        v = self.fc_v(v).view(batch, len_v, self.head, self.d_v).permute(0,2,1,3)
        
        out = self.attn(q, k, v).permute(0,2,1,3).contiguous().view(batch, len_v, -1)

        out = self.dropout(self.fc_o(out))
        # out = self.fc_o(out)
        
        out = out + shortcut_v

        out = self.layer_norm(out)
        return out

class MLP(nn.Module):
    def __init__(self, d_feature, d_hidden, dropout=0.1):
        super().__init__()

        self.fc_1 = nn.Linear(d_feature, d_hidden)
        self.fc_2 = nn.Linear(d_hidden, d_feature)
        self.layer_norm = nn.LayerNorm(d_feature, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut_x = x
        out = self.fc_1(x)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.dropout(out) + shortcut_x
        out = self.layer_norm(out)

        return out

class Encoderlayer(nn.Module):
    def __init__(self, d_feature, d_hidden, d_k, d_v, head, dropout=0.1):
        super(Encoderlayer, self).__init__()

        self.multihead = Multihead(d_feature, d_k, d_v, head)
        self.mlp = MLP(d_feature, d_hidden)

    def forward(self,x):
        out = self.multihead(x,x,x)
        out = self.mlp(out)
        return out

class Encoder(nn.Module):
    def __init__(self, layer, d_feature, d_hidden, d_k, d_v, head, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            Encoderlayer(d_feature, d_hidden, d_k, d_v, head, dropout)
            for _ in range(layer)])

    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        return x


   
if __name__ == "__main__":
    x = cv2.imread('./img/imgs/zebra155.jpg')
    x = cv2.resize(x, (640,640),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x,0).permute(0,3,1,2)
    
    emb = PatchEmbed()
    model = Encoder(3, 512, 512, 128, 128, 8)
    x_embed = emb(x)
    out = model(x_embed)
    print(out.shape)

