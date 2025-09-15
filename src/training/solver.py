import torch
import os
import tqdm
import numpy as np
from ..utils.utils import save_checkpoint, pca_inverse, load_checkpoint

class Trainer:
    def __init__(self, configs, model, train_loader, val_loader, test_loader=None):
        self.device = configs["training"]["device"]
        self.model = model.to(self.device)
        self.net = model.net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.configs = configs

        # Optimizer & Criterion (Default: Adam & MSELoss)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(configs["training"]["lr"]),
            weight_decay=float(configs["training"]["weight_decay"]),
        )
        self.criterion = torch.nn.MSELoss()

        # Scheduler (optional)
        self.scheduler = None
        if configs["training"].get("scheduler", None) == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=configs["training"]["num_epochs"]
            )

        # Working Directorys
        self.save_dir = configs["dir"]["save_dir"]
        self.checkpoint_dir = configs["dir"]["checkpoint_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for x, y, embedding in tqdm.tqdm(self.train_loader, desc=f"Train {epoch}"):
            x, y, embedding = x.to(self.device), y.to(self.device), embedding.to(self.device)
            pred, noise = self.model(x, y, embedding)  
            loss = self.criterion(pred, noise)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return running_loss / len(self.train_loader)

    @torch.no_grad() 
    def validate_epoch(self, epoch):
        # Validate the performance on predicting noise (or the denoised x_0)
        if self.val_loader is None:
            return None
        self.model.eval()
        running_loss = 0.0

        for x, y, embedding in tqdm.tqdm(self.val_loader, desc=f"Val {epoch}"):
            x, y, embedding = x.to(self.device), y.to(self.device), embedding.to(self.device)
            pred, noise = self.model(x, y, embedding)  
            loss = self.criterion(pred, noise)
            running_loss += loss.item()

        return running_loss / len(self.val_loader)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model.pt")
        self.model, _, _, _ = load_checkpoint(self.model, checkpoint_path, device=self.device) # model_state, optimizer, epoch, val_loss
        running_loss = 0.0
        all_pred = []
        all_target = []

        for x, y, embedding in tqdm.tqdm(self.test_loader, desc="Testing"):
            x = x.to(self.device)
            y = y.to(self.device)
            embedding = embedding.to(self.device)
            y_pred = self.model.reverse(x, embedding)
            running_loss += self.criterion(y_pred, y).item()
            all_pred.append(y_pred.cpu())
            all_target.append(y.cpu())

        all_pred = torch.cat(all_pred, dim=0)
        all_target = torch.cat(all_target, dim=0)
        test_loss = running_loss / len(self.test_loader)
        print(f"Low-rank Diffusion Test Loss: {test_loss:.4f}")
        # Inverse PCA transformation
        pca_model_path = os.path.join(self.configs["dir"]["save_dir"], "pca_Y.pkl")
        all_pred_reconstructed = pca_inverse(all_pred, pca_model_path)

        all_pred_reconstructed = all_pred_reconstructed.cpu().numpy()
        all_pred = all_pred.cpu().numpy()
        all_target = all_target.cpu().numpy()
        save_dir = self.configs["dir"]["pred_dir"]
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "test_outputs.npz"),
            pred_target=all_pred_reconstructed,
            pred_pc=all_pred,
            Y_pc=all_target,
        )
    
    def fit(self):
        num_epochs = self.configs["training"]["num_epochs"]
        save_every = self.configs["training"].get("save_every", 10)
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}" +
                  (f" | Val Loss: {val_loss:.4f}" if val_loss else ""))

            # save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.checkpoint_dir, f"best_model.pt")
                save_checkpoint(self.model, save_path, epoch, self.optimizer, best_val_loss)
            if epoch % save_every == 0:
                save_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pt")
                save_checkpoint(self.model, save_path, epoch, self.optimizer, val_loss)

if __name__ == '__main__':
    pass
                