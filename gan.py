import torch
from torch import autograd
import torch.nn as nn
from tqdm import tqdm
from plots import show_imgs, plot_stats
import os

class GAN(nn.Module):
    def __init__(self,
                 gen, gen_opt, gen_input_fn, gen_loss_fn,  # Assumes loss_fns add rather than mean.
                 discrim, discrim_opt, discrim_loss_fn,
                 discrim_weight_clipping=False,
                 discrim_gradient_clipping=False,
                 discrim_gradient_penalty=False,
                 discrim_simple_gradient_penalty=False,
                 gp_weight=10.0):
        
        super().__init__()
        
        self.gen = gen
        self.gen_opt = gen_opt
        self.gen_input_fn = gen_input_fn
        self.gen_loss_fn = gen_loss_fn
        
        self.discrim = discrim
        self.discrim_opt = discrim_opt
        self.discrim_loss_fn = discrim_loss_fn
        self.discrim_weight_clipping = discrim_weight_clipping
        self.discrim_gradient_clipping = discrim_gradient_clipping
        self.discrim_gradient_penalty = discrim_gradient_penalty
        self.discrim_simple_gradient_penalty = discrim_simple_gradient_penalty
        self.gp_weight = gp_weight
        
        self.epoch = 1
        
    def get_extra_state(self):
        return {"epoch": self.epoch}
        
    def set_extra_state(self, state):
        self.epoch = state["epoch"]

    def save_checkpoint(self, path):
        torch.save({"state": self.state_dict()}, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state"])
    
    def _gradient_penalty(self, real_samples, fake_samples):
        # Based on:
        # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
        alpha = torch.rand((real_samples.size(0),1,1,1), device=real_samples.get_device())
        interpolated = (alpha * real_samples + (1-alpha) * fake_samples).requires_grad_(True)
        discrim_out = self.discrim(interpolated)
        grad_out = torch.ones_like(discrim_out)
        grads = autograd.grad(
            outputs=discrim_out,
            inputs=interpolated,
            grad_outputs=grad_out,
            create_graph=True,
            retain_graph=True)[0]
        grads = grads.reshape(real_samples.size(0), -1)
        gp = ((grads.norm(2, dim=1) - 1) ** 2).sum()  # Using sum instead of mean as everywhere else does.
        return gp
    
    def _simple_gradient_penalty(self, real_samples, fake_samples,
                                 discrim_scores_real, discrim_scores_fake):
        # https://lernapparat.de/improved-wasserstein-gan/
        # https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Semi-Improved_Training_of_Wasserstein_GAN.ipynb
        # This is useful as Pytorch doesn't support 2nd order deriv with attention.
        dist = ((real_samples - fake_samples) ** 2).sum(1) ** .5
        print(discrim_scores_real.size())
        print(discrim_scores_fake.size())
        print(dist.size())
        est = (discrim_scores_real - discrim_scores_fake).abs()/(dist + 1e-8)
        return ((1-est) ** 2).sum(0).view(1)
    
    def _train_batch(self, batch):
        self.gen.train()
        self.discrim.train()
        
        gen_out = self.gen(self.gen_input_fn(batch))
        discrim_scores_real = self.discrim(batch)
        discrim_scores_fake = self.discrim(gen_out)
        
        discrim_loss = self.discrim_loss_fn(discrim_scores_real, discrim_scores_fake)
        if self.discrim_gradient_penalty:
            discrim_loss += self._gradient_penalty(batch, gen_out) * self.gp_weight
        
        if self.discrim_simple_gradient_penalty:
            discrim_loss += self._simple_gradient_penalty(batch, gen_out,
                                                          discrim_scores_real,
                                                          discrim_scores_fake) * self.gp_weight
        
        self.discrim_opt.zero_grad()
        discrim_loss.backward(retain_graph=True)
        if self.discrim_gradient_clipping:
            nn.utils.clip_grad_value_(self.discrim.parameters(), 1.0)
            
        self.discrim_opt.step()
        
        if self.discrim_weight_clipping:
            with torch.no_grad():
                for param in self.discrim.parameters():
                    param.clamp_(-1, 1)
        
        gen_loss = self.gen_loss_fn(self.discrim(gen_out), batch, gen_out)
        
        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()
                
        return float(gen_loss), float(discrim_loss)
    
    def _train_epoch(self, dl_train, verbose):
        total_gen_loss, total_discrim_loss = 0, 0
        loop_display = tqdm if verbose else lambda x: x
        for batch in loop_display(dl_train):
            gen_loss, discrim_loss = self._train_batch(batch)
            total_gen_loss += gen_loss
            total_discrim_loss += discrim_loss
        
        return total_gen_loss, total_discrim_loss
    
    def validate(self, dl_val):
        self.gen.eval()
        self.discrim.eval()
        
        total_gen_loss, total_discrim_loss = 0, 0
        with torch.no_grad():
            for batch in dl_val:
                gen_out = self.gen(self.gen_input_fn(batch))
                discrim_scores_real = self.discrim(batch)
                discrim_scores_fake = self.discrim(gen_out)
                discrim_loss = self.discrim_loss_fn(discrim_scores_real, discrim_scores_fake)
                gen_loss = self.gen_loss_fn(discrim_scores_fake, batch, gen_out)
                total_gen_loss += gen_loss
                total_discrim_loss += discrim_loss
        
        return float(total_gen_loss), float(total_discrim_loss)
    
    def train_loop(self, dl_train, dl_val, checkpoint_directory, full_display_epochs, verbose):
        while True:
            train_loss_gen, train_loss_discrim = self._train_epoch(dl_train, verbose)
            train_loss_gen /= len(dl_train.dataset)
            train_loss_discrim /= len(dl_train.dataset)
            
            val_loss_gen, val_loss_discrim = self.validate(dl_val)
            val_loss_gen /= len(dl_val.dataset)
            val_loss_discrim /= len(dl_val.dataset)
            
            if verbose:
                print(f"Epoch {self.epoch}: ")
                print(f"    Generator Train Loss: {train_loss_gen:.3f}")
                print(f"      Generator Val Loss: {val_loss_gen:.3f}")
                print(f"Discriminator Train Loss: {train_loss_discrim:.3f}")
                print(f"  Discriminator Val Loss: {val_loss_discrim:.3f}")
            
            self.gen.train_losses.append(train_loss_gen)
            self.discrim.train_losses.append(train_loss_discrim)
            self.gen.val_losses.append(val_loss_gen)
            self.discrim.val_losses.append(val_loss_discrim)
            self.epoch += 1
            
            if not (self.epoch-1) % full_display_epochs:
                if not verbose:
                    print(f"Epoch {self.epoch-1}: ")  # In verbose, this is already printed.
                    
                plot_stats((self.gen.train_losses, self.gen.val_losses),
                           ("Generator Training Loss", "Generator Validation Loss"),
                           "Generator Loss Curves")
                plot_stats((self.discrim.train_losses, self.discrim.val_losses),
                           ("Discriminator Training Loss", "Discriminator Validation Loss"),
                           "Discriminator Loss Curves")
                show_imgs(self.gen(self.gen_input_fn(dl_train.dataset[:10])), "Generated Training Images")
                show_imgs(self.gen(self.gen_input_fn(dl_val.dataset[:10])), "Generated Validation Images")
                if checkpoint_directory:
                    self.save_checkpoint(os.path.join(checkpoint_directory, f"checkpoint-{self.epoch-1}.tar"))
                
        
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.train_losses = []
        self.val_losses = []
        
    def get_extra_state(self):
            return {"train_losses": self.train_losses, "val_losses": self.val_losses}
            
    def set_extra_state(self, state):
            self.train_losses = state["train_losses"]
            self.val_losses = state["val_losses"]
            

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.train_losses = []
        self.val_losses = []
        
    def get_extra_state(self):
            return {"train_losses": self.train_losses, "val_losses": self.val_losses}
            
    def set_extra_state(self, state):
            self.train_losses = state["train_losses"]
            self.val_losses = state["val_losses"]