from torch.utils.data import DataLoader
from load_image_dataset import ImageDataset
import torch
from model import VisionTransformer, device
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from config import ModelArgs
import matplotlib.pyplot as plt

BATCH_SIZE = 4

train_config = ModelArgs()
val_config = ModelArgs()
val_config.mode = 'valid'

train_dataset = ImageDataset(train_config)
val_dataset = ImageDataset(val_config)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=BATCH_SIZE)


@torch.no_grad()
def get_eval_loss(model):
    accumulate_loss = 0.0
    accumulate_n = 0
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for xb, yb in val_dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        loss = criterion(out.reshape(-1, val_config.num_classes), yb.reshape(-1))
        accumulate_loss += loss.item()
        accumulate_n += 1
    model.train()

    return accumulate_loss/accumulate_n




def train():
    model = VisionTransformer(train_config)
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)
    stepi = []
    step = 0
    lossi = []
    evali = []
    accumulate_loss = 0.0
    accumulate_n = 0
    for epoch in range(train_config.num_epochs):
        for xb, yb in train_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            accumulate_n +=1
            optimizer.zero_grad()
            out = model(xb)
            out = out.reshape(-1, train_config.num_classes)
            yb = yb.reshape(-1)
            loss = criterion(out, yb)
            accumulate_loss += loss.item()
            loss.backward()
            optimizer.step()
            if step%200==0:
                train_loss = accumulate_loss / accumulate_n
                accumulate_loss = 0.0
                accumulate_n = 0
                eval_loss = get_eval_loss(model)
                stepi.append(step)
                lossi.append(train_loss)
                evali.append(eval_loss)
                print(f"Epoch: {epoch}, step:- {step}, Loss:- {train_loss}, eval_loss:- {eval_loss}")
            step += 1
        file_name = f"{train_config.model_folder}/checkpoint_{step}.pth"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "global_step": step}, file_name)
    
    plt.figure(figsize=(10, 6))
    plt.plot(stepi, lossi, label='Training Loss', color='blue')
    plt.plot(stepi, evali, label='Validation Loss', color='orange')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    print("Plot saved as training_loss_plot.png")



if __name__ == '__main__':
    train()
