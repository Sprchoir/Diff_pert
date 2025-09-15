import yaml
from src.data.dataloader import get_dataloader
from src.models.diffusion import DiffusionModel
from src.training.solver import Trainer

# Configurations
with open("./Diff_pert/configs/config.yaml", "r") as f:
    configs = yaml.safe_load(f)

# Train A Diffusion Model
train_loader = get_dataloader(configs, split="train")
val_loader = get_dataloader(configs, split="val")
model = DiffusionModel(configs)
trainer = Trainer(configs, model, train_loader, val_loader)
print('Start Training......')
trainer.fit()
print('Training Done!')

# Test the model
test_loader = get_dataloader(configs, split="test")
trainer.test_loader = test_loader
print('Start Testing......')
trainer.test()
print("Done!")




