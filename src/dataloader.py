from torch.utils.data import DataLoader
from src.dataset import VoiceDataset
from src.utils import load_config

cfg = load_config("config/config.yaml")

train_dataset = VoiceDataset(cfg, mode="train")
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=False
)
