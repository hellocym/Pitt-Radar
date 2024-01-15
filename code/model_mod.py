import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def conv_block(in_chan, out_chan, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm3d(out_chan),
        nn.ReLU(inplace=True)
    )


def conv_stage(in_chan, out_chan):
    return nn.Sequential(
        conv_block(in_chan, out_chan),
        conv_block(out_chan, out_chan),
    )


class UNetPP_3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc00 = conv_stage(1, 16)
        self.enc10 = conv_stage(16, 32)
        self.enc20 = conv_stage(32, 64)
        self.enc30 = conv_stage(64, 128)
        self.enc40 = conv_stage(128, 128)
        self.pool = nn.MaxPool3d(2, 2)

        self.dec41 = conv_stage(256, 64)
        self.dec32 = conv_stage(64+32+64, 32)
        self.dec23 = conv_stage(32+16+16+32, 16)
        self.dec14 = conv_stage(16+16+16+16+16, 16)
        self.conv_out1 = nn.Conv3d(16, 1, 1)

        self.dec31 = conv_stage(64+128, 32)
        self.dec22 = conv_stage(32+16+32, 16)
        self.dec13 = conv_stage(16+16+16+16, 16)
        self.conv_out2 = nn.Conv3d(16, 1, 1)

        self.dec21 = conv_stage(32+64, 16)
        self.dec12 = conv_stage(16+16+16, 16)
        self.conv_out3 = nn.Conv3d(16, 1, 1)

        self.dec11 = conv_stage(16+32, 16)
        self.conv_out4 = nn.Conv3d(16, 1, 1)

        self.conv_out5 = nn.Conv3d(16, 1, 1)
        

    def forward(self, x):
        # x: [batch_size, 1, depth, height, width]
        # out: [batch_size, 1, depth, height, width]
        x10 = self.enc00(x)
        x20 = self.enc10(self.pool(x10))
        x30 = self.enc20(self.pool(x20))
        x40 = self.enc30(self.pool(x30))
        x50 = self.enc40(self.pool(x40))

        x41 = self.dec41(torch.cat((x40, F.upsample(x50, x40.size()[2:], mode='trilinear')), 1))
        x31 = self.dec31(torch.cat((x30, F.upsample(x40, x30.size()[2:], mode='trilinear')), 1))
        x21 = self.dec21(torch.cat((x20, F.upsample(x30, x20.size()[2:], mode='trilinear')), 1))
        x11 = self.dec11(torch.cat((x10, F.upsample(x20, x10.size()[2:], mode='trilinear')), 1))
        out1 = self.conv_out1(x10)
        
        x32 = self.dec32(torch.cat((x30, x31, F.upsample(x41, x31.size()[2:], mode='trilinear')), 1))
        x22 = self.dec22(torch.cat((x20, x21, F.upsample(x31, x21.size()[2:], mode='trilinear')), 1))
        x12 = self.dec12(torch.cat((x10, x11, F.upsample(x21, x11.size()[2:], mode='trilinear')), 1))
        out2 = self.conv_out2(x11)

        x23 = self.dec23(torch.cat((x20, x21, x22, F.upsample(x32, x22.size()[2:], mode='trilinear')), 1))
        x13 = self.dec13(torch.cat((x10, x11, x12, F.upsample(x22, x12.size()[2:], mode='trilinear')), 1))
        out3 = self.conv_out3(x12)

        x14 = self.dec14(torch.cat((x10, x11, x12, x13, F.upsample(x23, x13.size()[2:], mode='trilinear')), 1))
        out4 = self.conv_out4(x13)

        out5 = self.conv_out5(x14)

        
        # out = F.sigmoid(out)
        return (out1 + out2 + out3 + out4 + out5) / 5


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout),
                ),
                
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            # prenorm residual
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            x = x + ff(norm2(x))
        return x


class ViT_3D(nn.Module):
    def __init__(self, image_size, patch_size=(8,8,8), dim=252, pool='cls', channels=3, emb_dropout=0.1):
        super().__init__()
        image_height, image_width, image_depth = image_size
        patch_height, patch_width, patch_depth = patch_size
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0, 'Image must be divisible by the patch.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        print(f'num_patches: {num_patches}')
        down_factor = 2
        n_patches = int((image_size[0]/2**down_factor// patch_size[0]) * (image_size[1]/2**down_factor// patch_size[1]) * (image_size[2]/2**down_factor// patch_size[2]))
        print(f'n_patches: {n_patches}')
        # patch_dim = channels * patch_height * patch_width * patch_depth
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.Conv3d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2), # -> [batch_size, dim, num_patches]
            Rearrange('b d n -> b n d'), # -> [batch_size, num_patches, dim]
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth=12, heads=12, mlp_dim=3072, dropout=0.1)

        self.pool = pool
        self.to_latent = nn.Identity()
        # self.to_latent = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)


    def forward(self, x):
        x = self.to_patch_embedding(x) # [batch_size, channel, h, w, d] -> [batch_size, num_patches, dim]
        # print(x.shape)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # add position embedding
        # print(self.pos_embedding[:, :(n)].shape)
        # print(f'pe: {self.pos_embedding.shape}')
        x += self.pos_embedding
        # print(x.shape)
        x = self.dropout(x)

        x = self.transformer(x) # [batch_size, num_patches, dim], num_patches = h*w*d/(ph*pw*pd)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, needs_pooling=True):
        super().__init__()
        self.needs_pooling = needs_pooling

        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        self.instance_norm1 = nn.InstanceNorm3d(out_chan)
        self.conv2 = nn.Conv3d(out_chan, out_chan, kernel_size=3, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)
        self.instance_norm2 = nn.InstanceNorm3d(out_chan)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.instance_norm1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.instance_norm2(x)
        if not self.needs_pooling:
            return x
        x = self.pool(x)
        return x


class ConvUpBlock(nn.Module):
    def __init__(self, in_chan, skip_chan, out_chan, stride=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(in_chan+skip_chan, out_chan, kernel_size=3, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        self.instance_norm1 = nn.InstanceNorm3d(out_chan)
        self.conv2 = nn.Conv3d(out_chan, out_chan, kernel_size=3, padding=1)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)
        self.instance_norm2 = nn.InstanceNorm3d(out_chan)


    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.instance_norm1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.instance_norm2(x)
        return x



class ViT_V_Net(nn.Module):
    '''
    ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration
    https://arxiv.org/abs/2104.06468
    '''
    def __init__(self, image_size, channels, patch_size, emb_dropout=0., dim=252):
        super().__init__()
        self.img_size = image_size
        self.patch_size = patch_size
        self.down_factor = 2


        self.conv1 = ConvBlock(channels, 16, needs_pooling=False)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 32)
        self.ViT = ViT_3D(image_size=image_size, patch_size=patch_size, dim=dim, channels=32, emb_dropout=emb_dropout)
        self.conv4 = nn.Conv3d(dim, 512, kernel_size=3, padding=1, bias=False)
        self.leaky_relu4 = nn.LeakyReLU(inplace=True)
        self.conv5 = ConvBlock(512, 64, needs_pooling=False)
        self.leaky_relu5 = nn.LeakyReLU(inplace=True)
        
        self.convup1 = ConvUpBlock(96, 32, 48)
        self.convup2 = ConvUpBlock(48, 32, 32)
        self.convup3 = ConvUpBlock(32, 32, 32)
        self.convup4 = ConvUpBlock(32, 16, 16)

        self.convout = nn.Conv3d(16, 3, kernel_size=3, padding=1, bias=False)


    def forward(self, x):
        down1 = self.conv1(x)
        # print(down1.shape)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        print(down3.shape)
        x = self.ViT(down3) # [batch_size, num_patches, dim]
        print(down3.shape)
        # reshape from [batch_size, num_patches, dim] to [batch_size, h, w, d]
        B, N, D = x.shape
        # print(x.shape)
        h, w, d = (self.img_size[0]//2**self.down_factor//self.patch_size[0]), (self.img_size[1]//2**self.down_factor//self.patch_size[1]), (self.img_size[2]//2**self.down_factor//self.patch_size[2])
        # print(h, w, d)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = x.contiguous().view(B, D, h, w, d)
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.conv5(x)
        x = self.leaky_relu5(x)
        # upsample x by 2
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        print(down3.shape)
        # downsample down3 by 4
        down3_4 = F.interpolate(down3, scale_factor=0.25, mode='trilinear', align_corners=True)
        x = torch.cat((x, down3_4), dim=1)
        # downsample down3 by 2
        down3_2 = F.interpolate(down3, scale_factor=0.5, mode='trilinear', align_corners=True)
        x = self.convup1(x, down3_2)
        x = self.convup2(x, down3)
        x = self.convup3(x, down2)
        x = self.convup4(x, down1)
        x = self.convout(x)
        return x
        

if __name__ == '__main__':
    model = ViT_V_Net(image_size=(64, 64, 64), channels=2, patch_size=(8, 8, 8), emb_dropout=0.1)
    x = torch.randn(2, 2, 64, 64, 64)
    out = model(x)
    print(out.shape)