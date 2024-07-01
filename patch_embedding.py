import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, height, width, in_dim, out_dim, patch_size):
        super().__init__()
        assert not height % patch_size, f"Height must be a multiple of patch size - got {height}."
        assert not width % patch_size, f"Width must be a multiple of patch size - got {width}."
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches_height = height // patch_size
        self.num_patches_width = width // patch_size 
        self.num_patches = self.num_patches_height * self.num_patches_width
        self.embed_dim = out_dim
        
    def forward(self, x):
        return self.conv(x).permute(0, 2, 3, 1)  # Move channels last.