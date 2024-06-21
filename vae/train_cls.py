from models import EyeClassifier
import os
import wandb
from utils.dataset import get_image_list_l,ClassifierDataset 
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
import numpy as np
import argparse
import time

class ClassifierTrainer(object):

    def __init__(self, args):

        self.device = args.device

        self.pretrained_vae = args.pretrained_vae
        self.model = EyeClassifier(args.latent_dim, self.pretrained_vae)

        self.model.to(self.device)
        self.cls_threshold = args.cls_threshold
        self.labeled_dir = args.labeled_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.num_iters = args.num_iters
        self.saving_dir = os.path.join(args.output_dir, 'cls', 'cls_{}'.format(timestamp))

    def compute_eer(self, labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        eer_index = np.argmin(np.abs(fnr - fpr))
        eer = (fpr[eer_index] + fnr[eer_index]) / 2
        thresh = thresholds[eer_index]
        return eer, thresh
    
    def evaluate(self,model, loader, device, ckpt_path=None):
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path))

        model.train(False)
        val_loss = 0.0
        correct = 0
        total = 0
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = F.binary_cross_entropy(outputs, labels)
                val_loss += loss.item()

                #probs = torch.softmax(outputs, dim=1)
                preds = outputs > self.cls_threshold
                #preds = torch.argmax(probs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                val_labels += labels.cpu().numpy().tolist()
                #print (outputs.shape)
                val_probs += outputs.cpu().numpy().tolist()

        eer, _ = self.compute_eer(val_labels, val_probs)
        accuracy = correct / total

        return val_loss, accuracy, eer
    
    def train(self):
        run = wandb.init(project = 'CLS_VAE')
        X_train,X_val,y_train,y_val = get_image_list_l(self.labeled_dir)

        train_dataset = ClassifierDataset(X_train, y_train)
        val_dataset = ClassifierDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=self.num_workers)

        all_params = list(self.model.parameters())
        optimizable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(optimizable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        step_idx = 0
        min_val_eer = np.inf
        while step_idx < self.num_iters:
            train_loss = 0.0
            for idx_batch, (image_batch, target_batch) in enumerate(train_loader):
                self.model.train()
                image_batch = image_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                pred_batch = self.model(image_batch)
                optimizer.zero_grad()
                loss = F.binary_cross_entropy(pred_batch, target_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if step_idx % 100 == 0:
                    val_loss, val_accuracy, val_eer = self.evaluate(self.model, val_loader, self.device)

                    run.log({"train_loss": loss, "val_loss": val_loss, 'val_accuracy': val_accuracy, 'val_eer': val_eer})



                    print(
                        f'Iteration {step_idx}  '
                        f'---- Train Loss: {loss.item():.5f} ---- | '
                        f'---- Val loss: {val_loss:.5f} ---- | '
                        f'---- Val accuracy: {val_accuracy:.5f} ---- | '
                        f'---- Val EER: {val_eer:.4f}')
                step_idx += 1
        wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', default='outputs_cls', help='path to save training files', type=str)
    parser.add_argument('--latent_dim', default=64, help='dimension of the latent space', type=int)
    parser.add_argument('--batch_size', default=64, help='batch size for training', type=int)
    parser.add_argument('--num_workers', default=4, help='number of workers for dataloader', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='learning rate for optimizer', type=float)
    parser.add_argument('--weight_decay', default=2e-5, help='weight decay for optimizer', type=float)
    parser.add_argument('--num_iters', default=10000, help='number of iterations for training', type=int)
    parser.add_argument('--device', default='cuda:0', help='device to use', type=str)
    parser.add_argument('--pretrained_vae', '-pretrain_vae', default=None,
                        help='pretrained VAE model', type=str)
    parser.add_argument('--cls_threshold', '-cls_thresh', default=0.5,
                        help='threshold for sigmoid output', type=float)
    parser.add_argument('--labeled_dir', default=None, help='path to labeled dir', type=str)

    args = parser.parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cls_trainer = ClassifierTrainer(args)
    cls_trainer.train()