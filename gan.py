import torch
import torch.nn as nn
from tqdm import tqdm
from plots import show_imgs, plot_stats
import os

class GAN(nn.Module):
    def __init__(self,
                 gen, gen_opt, gen_input_fn, gen_loss_fn,  # Assumes loss_fns add rather than mean.
                 discrim, discrim_opt, discrim_loss_fn):
        
        super().__init__()
        
        self.gen = gen
        self.gen_opt = gen_opt
        self.gen_input_fn = gen_input_fn
        self.gen_loss_fn = gen_loss_fn
        
        self.discrim = discrim
        self.discrim_opt = discrim_opt
        self.discrim_loss_fn = discrim_loss_fn
        
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
        
    def _train_batch(self, batch):
        self.gen.train()
        self.discrim.train()
        
        gen_out = self.gen(self.gen_input_fn(batch))
        discrim_scores_real = self.discrim(batch)
        discrim_scores_fake = self.discrim(gen_out)
        
        discrim_loss = self.discrim_loss_fn(discrim_scores_real, discrim_scores_fake)
        
        self.discrim_opt.zero_grad()
        discrim_loss.backward(retain_graph=True)
        self.discrim_opt.step()
        
        gen_loss = self.gen_loss_fn(self.discrim(gen_out))
        
        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()
                
        return float(gen_loss), float(discrim_loss)
    
    def _train_epoch(self, dl_train):
        total_gen_loss, total_discrim_loss = 0, 0
        for batch in tqdm(dl_train):
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
                gen_loss = self.gen_loss_fn(discrim_scores_fake)
                total_gen_loss += gen_loss
                total_discrim_loss += discrim_loss
        
        return float(total_gen_loss), float(total_discrim_loss)
    
    def train_loop(self, dl_train, dl_val, checkpoint_directory):
        while True:
            print(f"Epoch {self.epoch}: ")
            train_loss_gen, train_loss_discrim = self._train_epoch(dl_train)
            train_loss_gen /= len(dl_train.dataset)
            train_loss_discrim /= len(dl_train.dataset)
            
            val_loss_gen, val_loss_discrim = self.validate(dl_val)
            val_loss_gen /= len(dl_val.dataset)
            val_loss_discrim /= len(dl_val.dataset)
            
            self.gen.train_losses.append(train_loss_gen)
            self.discrim.train_losses.append(train_loss_discrim)
            self.gen.val_losses.append(val_loss_gen)
            self.discrim.val_losses.append(val_loss_discrim)
            self.epoch += 1
            
            if not (self.epoch-1) % 10:
                plot_stats((self.gen.train_losses, self.gen.val_losses, self.discrim.train_losses, self.discrim.val_losses),
                           ("Generator Training Loss", "Generator Validation Loss",
                            "Discriminator Training Loss", "Discriminator Validation Loss"),
                           "Loss Curves")
                show_imgs(self.gen(self.gen_input_fn(dl_train.dataset[:10])), "Generated Images")
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