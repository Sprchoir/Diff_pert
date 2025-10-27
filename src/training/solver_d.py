import os
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from ..utils.utils import save_checkpoint, load_checkpoint, inverse_transform
from ..utils.metrics import decoder_loss, pearson_r2_after_control, mse_all_genes, mse_top_DE_genes

class DecoderTrainer:
    def __init__(self, configs, model, model_re, train_loader=None, val_loader=None, test_loader=None):
        self.device = configs["training"]["device"]
        self.model = model.to(self.device)
        self.model_re = model_re.to(self.device)
        self.configs = configs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tiny_threshold = float(configs["decoder"]["tiny_threshold"])

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(configs["decoder"]["lr"]),
            weight_decay=float(configs["decoder"]["weight_decay"]),
        )
        self.optimizer_re = torch.optim.AdamW(
            model_re.parameters(),
            lr=float(configs["decoder"]["lr"]),
            weight_decay=float(configs["decoder"]["weight_decay"]),
        )
        
        # Scheduler: warmup + cosine
        self.scheduler = None
        num_epochs = configs["decoder"]["num_epochs"]
        num_steps = num_epochs * len(train_loader) if train_loader is not None else 1
        warmup_steps = int(0.03 * num_steps)  # 3% warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
            # cosine annealing to 0.3*lr
            return 0.3 + 0.7 * 0.5 * (1 + np.cos(np.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Working dirs
        self.save_dir = configs["dir"]["save_dir"]
        self.checkpoint_dir = configs["dir"]["checkpoint_dir"]
        self.pred_dir = configs["dir"]["pred_dir"]
        self.stats_file = os.path.join(self.pred_dir, f"stats.pkl")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss_decoder = 0.0

        # Train the main decoder first
        for Y_low, Y_full, idx, embeddings in tqdm.tqdm(self.train_loader, desc=f"Train_decoder {epoch}"):
            Y_low = Y_low.to(self.device)
            Y_full = Y_full.to(self.device)
            embeddings = embeddings.to(self.device)
            Y_pred = self.model(Y_low)
            # Y_pred = torch.where(Y_pred < self.tiny_threshold, torch.zeros_like(Y_pred), Y_pred)
            loss_decoder = decoder_loss(Y_pred, Y_full)
            self.optimizer.zero_grad()
            loss_decoder.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            running_loss_decoder += loss_decoder.item()
        loss_decoder = running_loss_decoder/len(self.train_loader)

        return loss_decoder
    
    def train_re_epoch(self, epoch, model):
        self.model_re.train()
        running_loss_re = 0.0

        for Y_low, Y_full, idx, embeddings in tqdm.tqdm(self.train_loader, desc=f"Train_Lasso {epoch}"):
            Y_low = Y_low.to(self.device)
            Y_full = Y_full.to(self.device)
            with torch.no_grad():
              Y_pred = model(Y_low)
            Y_pred = self.model_re(Y_pred)
            loss_re = self.model_re.loss_fn(Y_pred, Y_full)
            self.optimizer_re.zero_grad()
            loss_re.backward()
            self.optimizer_re.step()
            with torch.no_grad():
                self.model_re.linear.weight.data = self.model_re.linear.weight.data * (torch.abs(self.model_re.linear.weight.data) > self.model_re.lambda_l1 * self.optimizer_re.param_groups[0]["lr"])
            running_loss_re += loss_re.item()
        loss_re = running_loss_re/len(self.train_loader)

        return loss_re

    @torch.no_grad()
    def validate_epoch(self, epoch, stage):
        self.model.eval()
        self.model_re.eval()
        running_loss = 0.0

        for Y_low, Y_full, idx, embeddings in tqdm.tqdm(self.val_loader, desc=f"Val {epoch}"):
            Y_low = Y_low.to(self.device)
            Y_full = Y_full.to(self.device)
            embeddings = embeddings.to(self.device)
            Y_pred = self.model(Y_low)
            # Y_pred = torch.where(Y_pred < self.tiny_threshold, torch.zeros_like(Y_pred), Y_pred)
            if stage == "regularization":
                Y_pred = self.model_re(Y_pred)
            loss = F.mse_loss(Y_pred, Y_full)
            running_loss += loss.item()

        return running_loss/len(self.val_loader)

    def fit(self):
        num_epochs = self.configs["decoder"]["num_epochs"]
        save_every = self.configs["training"].get("save_every", 20)
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs+1):
            train_loss_1 = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch, "decoder")
            print(f"[Epoch {epoch}] Train Loss (Decoder)={train_loss_1:.4f} | Val Loss={val_loss:.4f}")

            # Save checkpoint of best decoder model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_decoder.pt")
                save_checkpoint(self.model, checkpoint_path, epoch, self.optimizer, best_val_loss)
            if epoch % save_every == 0:
                save_path_1 = os.path.join(self.checkpoint_dir, f"decoder_epoch_{epoch}.pt")
                save_checkpoint(self.model, save_path_1, epoch, self.optimizer, val_loss)
        
        self.model, _, _, _ = load_checkpoint(self.model, checkpoint_path, device=self.device)
        self.model.eval()
        best_val_loss = float("inf")
        for epoch in range(1, num_epochs+1):
            train_loss_2 = self.train_re_epoch(epoch, self.model)
            val_loss = self.validate_epoch(epoch, "regularization")
            print(f"[Epoch {epoch}] Train Loss (regularization)={train_loss_2:.4f} | Val Loss={val_loss:.4f}")

            # Save checkpoint of the best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_re.pt")
                save_checkpoint(self.model_re, checkpoint_path, epoch, self.optimizer_re, best_val_loss)
            if epoch % save_every == 0:
                save_path_2 = os.path.join(self.checkpoint_dir, f"re_epoch_{epoch}.pt")
                save_checkpoint(self.model_re, save_path_2, epoch, self.optimizer_re, val_loss)
    
    @torch.no_grad()
    def apply(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"best_decoder.pt")
        checkpoint_path_2 = os.path.join(self.checkpoint_dir, f"best_re.pt")
        model, _, _, _ = load_checkpoint(self.model, checkpoint_path, device=self.device)
        model_re, _, _, _ = load_checkpoint(self.model_re, checkpoint_path_2, device=self.device)
        running_loss = 0.0
        all_pred = []
        all_target = []

        for Y_low, Y_full, idx, embeddings in tqdm.tqdm(self.test_loader, desc="Decode"):
            Y_low = Y_low.to(self.device)
            Y_full = Y_full.to(self.device)
            embeddings = embeddings.to(self.device)
            Y_pred = model(Y_low)
            Y_pred = model_re(Y_pred)
            # Y_pred = torch.where(Y_pred < self.tiny_threshold, torch.zeros_like(Y_pred), Y_pred)
            loss = F.mse_loss(Y_pred, Y_full)
            running_loss += loss.item()
            Y_pred = inverse_transform(Y_pred, self.stats_file, idx, "test")
            Y_true = inverse_transform(Y_full, self.stats_file, idx, "test")
            all_pred.append(Y_pred.cpu())
            all_target.append(Y_true.cpu())
        all_pred = torch.cat(all_pred, dim=0).numpy()
        all_target = torch.cat(all_target, dim=0).numpy()
        print(f"Decoder Loss={running_loss/len(self.test_loader):.4f}")

        save_dir = self.configs["dir"]["pred_dir"]
        np.savez(
            os.path.join(save_dir, "Final_outputs.npz"),
            pred = all_pred,
            true = all_target
        )
        print("Decoding Done!")
        
        # Metrics
        Ori = np.load(os.path.join(self.configs["dir"]["pred_dir"], "XY_ori.npz"))
        X = Ori["X_ori"][:all_pred.shape[0]]
        r2 = pearson_r2_after_control(X, all_target, all_pred)
        mse_overall = mse_all_genes(all_target, all_pred)
        mse_topDE = mse_top_DE_genes(X, all_target, all_pred)
        print(f"Overall MSE:{mse_overall} | Top 20 DE MSE:{mse_topDE} | Pearson Correlation:{r2}.")


