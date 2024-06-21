import torch.nn as nn
from torchvision import transforms, datasets, models
import torch
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.model_ft = models.efficientnet_b4(weights="EfficientNet_B4_Weights.DEFAULT")
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
    

class OpenEyesClassifier:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        self.model = self.load_model()

    def load_model(self):
        model = Net()
        model.load_state_dict(torch.load(self.ckpt_path))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, inpIm):
        img = Image.open(inpIm)
        img = img.convert('RGB')
        img = self.transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)

        with torch.no_grad():
            output = self.model(img)
            probs = torch.softmax(output, dim=1)
            is_open_score = probs[:, 1].cpu().numpy()

        return is_open_score


        

