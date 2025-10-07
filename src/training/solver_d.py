import os
import joblib
import tqdm
import torch
import numpy as np
from ..utils.utils import save_checkpoint, load_checkpoint
from ..utils.metrics import decoder_loss, pearson_r2_after_control, mse_all_genes, mse_top_DE_genes

class DecoderTrainer:
    def __init__(self, configs, model, train_loader=None, val_loader=None, test_loader=None):
        self.device = configs["training"]["device"]
        self.model = model.to(self.device)
        self.configs = configs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.l1_lambda = float(configs["decoder"].get("l1_lambda", 1e-5))
        self.tiny_threshold = float(configs["decoder"]["tiny_threshold"])

        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
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
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        l1_lambda = float(self.configs["decoder"]["l1_lambda"])
        for Y_low, Y_full in tqdm.tqdm(self.train_loader, desc=f"Train {epoch}"):
            Y_low = Y_low.to(self.device)
            Y_full = Y_full.to(self.device)

            Y_pred = self.model(Y_low)
            loss = decoder_loss(Y_pred, Y_full, l1_lambda)
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()

        return running_loss/len(self.train_loader)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0

        for Y_low, Y_full in tqdm.tqdm(self.val_loader, desc=f"Val {epoch}"):
            Y_low = Y_low.to(self.device)
            Y_full = Y_full.to(self.device)
            Y_pred = self.model(Y_low)
            loss = decoder_loss(Y_pred, Y_full)
            running_loss += loss.item()

        return running_loss/len(self.val_loader)

    def fit(self):
        num_epochs = self.configs["decoder"]["num_epochs"]
        save_every = self.configs["training"].get("save_every", 20)
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs+1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            print(f"[Epoch {epoch}] Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
            # Save checkpoint
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.checkpoint_dir, "best_decoder.pt")
                save_checkpoint(self.model, save_path, epoch, self.optimizer, best_val_loss)
            if epoch % save_every == 0:
                save_path = os.path.join(self.checkpoint_dir, f"decoder_epoch_{epoch}.pt")
                save_checkpoint(self.model, save_path, epoch, self.optimizer, val_loss)
    
    @torch.no_grad()
    def apply(self):
        self.model.eval()
        checkpoint_path = os.path.join(self.checkpoint_dir, f"best_decoder.pt")
        self.model, _, _, _ = load_checkpoint(self.model, checkpoint_path, device=self.device)
        running_loss = 0.0
        all_pred = []
        all_target = []
        for Y_low, Y_full in tqdm.tqdm(self.test_loader, desc="Decode"):
            Y_low = Y_low.to(self.device)
            Y_full = Y_full.to(self.device)
            Y_pred = self.model(Y_low)
            loss = decoder_loss(Y_pred, Y_full)
            running_loss += loss.item()
            all_pred.append(Y_pred.cpu())
            all_target.append(Y_full.cpu())
        all_pred = torch.cat(all_pred, dim=0).numpy()
        all_target = torch.cat(all_target, dim=0).numpy()
        print(f"Decoder Loss={running_loss/len(self.test_loader):.4f}")
        save_dir = self.configs["dir"]["pred_dir"]
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "Decoder_outputs.npz"),
            pred = all_pred,
            true = all_target
        )
        print("Decoding Done!")

        # Post-processing
        stats_file = os.path.join(self.pred_dir, f"stats.pkl")
        stats = joblib.load(stats_file)
        Y = all_pred
        # Y = all_pred * stats["std_Y"] + stats["mean_Y"]
        Y = np.expm1(Y)
        # Y = Y / stats["norm_sum"] * stats["lib_size_raw_Y"][:Y.shape[0]][:, None]
        Y_pred = np.where(Y < self.tiny_threshold, np.zeros_like(Y), Y)
        
        # Metrics
        Ori = np.load(os.path.join(self.configs["dir"]["pred_dir"], "XY_ori.npz"))
        Y_true = Ori["Y_ori"][:Y_pred.shape[0]]
        X = Ori["X_ori"][:Y_pred.shape[0]]
        r2 = pearson_r2_after_control(X, Y_true, Y_pred)
        mse_overall = mse_all_genes(Y_true, Y_pred)
        mse_topDE = mse_top_DE_genes(X, Y_true, Y_pred)
        print(f"Overall MSE:{mse_overall} | Top 20 DE MSE:{mse_topDE} | Pearson Correlation:{r2}.")
        np.savez(
            os.path.join(save_dir, "Final_outputs.npz"),
            pred = Y_pred,
            true = Y_true,
            X = X
        )