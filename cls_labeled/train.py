import os
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models
import wandb
from torchvision.transforms import ToTensor,Compose, CenterCrop, Normalize, RandomResizedCrop

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

BATCH_SIZE = 32


dataset = torchvision.datasets.ImageFolder(root='/home/lukavetoshkin/VisionLabs/cls_labeled/eyes_dataset_labeled', transform=train_transform)
train_set,val_set = torch.utils.data.random_split(dataset, [0.85, 0.15])

train_loader = DataLoader(train_set, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,  
                          num_workers=0)

val_loader = DataLoader(val_set, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        num_workers=0) 




# ________TRAIN_FUNCTION_______________________________________________________________________________

                    
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.argmin(np.abs(fnr - fpr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    thresh = thresholds[eer_index]
    return eer, thresh
    

def evaluate(model, loader, criterion, device, ckpt_path=None):
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_labels = []
    val_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            val_labels += labels.cpu().numpy().tolist()
            val_probs += probs[:, 1].cpu().numpy().tolist()

    eer, _ = compute_eer(val_labels, val_probs)
    accuracy = correct / total

    return val_loss, accuracy, eer


def train(model, epoch_num, optimizer, criterion, train_loader, val_loader, ckpt_save_path, lr_scheduler=None):
    
    wandb.watch(model, criterion, log="all", log_freq=1)
    min_val_eer = np.inf

    for epoch in range(epoch_num):

        train_loss = 0.0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        val_loss, val_accuracy, val_eer = evaluate(model, val_loader, criterion, device)

        wandb.log({"epoch": epoch, "loss": val_loss, 'val_accuracy': val_accuracy, 'val_eer': val_eer})

        print(f"Epoch {epoch}: train loss = {train_loss:.4f}")
        print(f"val loss = {val_loss:.4f}, val accuracy = {val_accuracy:.4f}, val eer = {val_eer:.4f}")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if val_eer < min_val_eer or val_eer < 0.02:
            min_val_eer = val_eer
            torch.save(model.state_dict(), ckpt_save_path)
            print(f"Saving weights with val accuracy = {val_accuracy:.4f}, val eer = {val_eer:.4f}")

    return min_val_eer


# ________TRAIN_MODEL_______________________________________________________________________________

wandb.init(project="open_eyes")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.model_ft = models.efficientnet_b4(weights="EfficientNet_B4_Weights.DEFAULT")
        #print (self.model_ft)
        #self.model_ft.classifier[1] = nn.Linear(1280, 2)
        self.linear1 = nn.Linear(1000,256)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(256)
        self.out = nn.Linear(256,2)
    def forward(self, x):
        x = self.model_ft(x)
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.out(x)
        return x

model = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),  lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

train(model, 100, optimizer,criterion, train_loader, val_loader, '/home/lukavetoshkin/VisionLabs/cls_labeled/weights/effnet.pth', lr_scheduler=scheduler)