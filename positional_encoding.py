import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import itertools

from plots import plot_stats


class PositionalEncoding2d(nn.Module):
    """Should be inherited from. Define self.enc"""
    # Height, width, embed dimension.
    
    def forward(self, x):
        return self.enc + x  # Encodes whole image at once. Encoding is broadcasted to match batch dimension.
    
    def visualize(self):
        height = self.enc.size(0)
        width = self.enc.size(1)
        embed_d = self.enc.size(2)
        
        fig, axs = plt.subplots(nrows=embed_d//2, ncols=2, figsize=(10, embed_d // 4))
        fig.suptitle("Dimensions of Positional Encoding")
        for d in range(embed_d):
            ax = axs[d % (embed_d // 2), d * 2 // embed_d]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            im = ax.imshow(self.enc[:,:,d].detach().cpu(), vmin=-1, vmax=1, cmap="gray")
    
        fig.colorbar(im, ax=axs.ravel().tolist())
        plt.show()
        
        print("Distances along diagonal 5")
        print(nn.functional.cosine_similarity(self.enc[0, 0, :], self.enc[5, 5, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[5, 5, :], self.enc[10, 10, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[10, 10, :], self.enc[15, 15, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[10, 6, :], self.enc[15, 11, :], dim=0))
        print("")
       
        print("Distances along height 5")
        print(nn.functional.cosine_similarity(self.enc[0, 0, :], self.enc[5, 0, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[5, 0, :], self.enc[10, 0, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[10, 0, :], self.enc[15, 0, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[10, 6, :], self.enc[15, 6, :], dim=0))
        print("")
       
        print("Distances along width 5")
        print(nn.functional.cosine_similarity(self.enc[0, 0, :], self.enc[0, 5, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[0, 5, :], self.enc[0, 10, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[0, 10, :], self.enc[0, 15, :], dim=0))
        print(nn.functional.cosine_similarity(self.enc[10, 6, :], self.enc[10, 11, :], dim=0)) 

        first_vec = self.enc[0, 0, :]
        vec_dists = [1-nn.functional.cosine_similarity(first_vec, self.enc[x, x, :], dim=0).detach().cpu() for x in range(min(height, width))]
        plot_stats((vec_dists,), ("Cosine distance",), "Distance From (0, 0)", xlabel="Coordinate (x, x)")
    
        vec_dists = [1-nn.functional.cosine_similarity(first_vec, self.enc[0, x, :], dim=0).detach().cpu() for x in range(width)]
        plot_stats((vec_dists,), ("Cosine distance",), "Distance From (0, 0)", xlabel="Coordinate (0, x)")
    
        vec_dists = [1-nn.functional.cosine_similarity(first_vec, self.enc[x, 0, :], dim=0).detach().cpu() for x in range(height)]
        plot_stats((vec_dists,), ("Cosine distance",), "Distance From (0, 0)", xlabel="Coordinate (x, 0)")
    
        fig, axs = plt.subplots(nrows=height, ncols=width, figsize=(width, height))
        fig.suptitle("Cosine Similarity (patch is one, pixel is other)")
    
        for x, y in itertools.product(range(height), range(width)):
            ax = axs[x, y]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    
            ax.imshow(nn.functional.cosine_similarity(self.enc[x, y, :], self.enc[:, :, :], dim=2).detach().cpu(), vmin=-1, vmax=1, cmap="gray")

        plt.show()
    

class SinusoidalPositionalEncoding2d(PositionalEncoding2d):
    def __init__(self, height, width, embed_dim, scale, trainable=False):
        super().__init__()
        assert embed_dim % 4 == 0, f"Embed dimension must be a multiple of 4 - got {embed_dim}."
        if trainable:
            self.enc = nn.Parameter(torch.empty((height, width, embed_dim)))  # Parameter makes it trainable. Register buffer is not.
        else:
            self.register_buffer("enc", torch.empty((height, width, embed_dim)))
                                 
        with torch.no_grad():
            for d in range(embed_dim):
                d_shift = d if d < embed_dim // 2 else d - embed_dim // 2
                if d % 2:
                    denom = math.exp(2*d_shift / embed_dim * math.log(width*scale))  # Use logspace to prevent overflow.
                    if d < embed_dim // 2:
                        self.enc[:,:,d] = (torch.arange(width) / denom).sin().view(1, -1).expand(height, -1)
                    else:
                        self.enc[:,:,d] = (torch.arange(height) / denom).cos().view(-1, 1).expand(-1, width)
                else:
                    denom = math.exp(2*d_shift / embed_dim * math.log(height*scale))
                    if d < embed_dim // 2:
                        self.enc[:,:,d] = (torch.arange(width) / denom).cos().view(1, -1).expand(height, -1)
                    else:
                        self.enc[:,:,d] = (torch.arange(height) / denom).sin().view(-1, 1).expand(-1, width)
                        

class LearnedPositionalEncoding2d(PositionalEncoding2d):
    def __init__(self, height, width, embed_dim):
        super().__init__()
        self.enc = nn.Parameter(torch.empty((height, width, embed_dim)))
        nn.init.normal_(self.enc)
        
class PositionalEncoding1d(nn.Module):
    """Should be inherited from. Define self.enc"""
    # Sequence length, embed dimension.
    
    def forward(self, x):
        return self.enc + x  # Encodes whole image at once. Encoding is broadcasted to match batch dimension.

class LearnedPositionalEncoding1d(PositionalEncoding1d):
    def __init__(self, length, embed_dim):
        super().__init__()
        self.enc = nn.Parameter(torch.empty((length, embed_dim)))
        nn.init.normal_(self.enc)