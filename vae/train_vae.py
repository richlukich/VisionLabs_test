from models import VanillaVAE, VAE
import wandb
import os
import time
from utils.dataset import get_image_list_u
import torch
import torch.nn.functional as F
import argparse

class VaeTrainer(object):

    def __init__(self, args):

        self.device = args.device
        #self.vae = VanillaVAE(1, args.latent_dim)
        self.vae = VAE(args.latent_dim)
        self.vae.to(self.device)
        self.beta = args.beta
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.learning_rate = args.learning_rate
        self.num_iters = args.num_iters
        self.saving_dir = os.path.join(args.output_dir, 'vae', 'vae_{}'.format(timestamp))
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)
        self.unlabeled_dir = args.unlabeled_dir
        
    def train(self):
        self.vae.train()
        run = wandb.init(project = 'VAE')
        image_list = get_image_list_u(self.unlabeled_dir)
        tensor_data = (1 / 255. * torch.tensor(image_list).float()).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(tensor_data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                   shuffle=True, num_workers=self.num_workers)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.learning_rate)

        step_idx = 0

        while step_idx < self.num_iters:
            for idx_batch, (batch,) in enumerate(train_loader):
                batch = batch.to(self.device)
                pred_batch, z, mu, log_var = self.vae(batch)
                optimizer.zero_grad()
                loss_recon = F.mse_loss(batch, pred_batch)
                loss_kld =  torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss = loss_recon + 0.001 * loss_kld
                loss.backward()
                optimizer.step()

                if step_idx % 100 == 0:
                    gt = []
                    pred = []
                    
                    for i in range (10):
                        gt.append(wandb.Image(batch[i].detach().cpu().numpy().reshape(24,24,1)))
                        pred.append(wandb.Image(pred_batch[i].detach().cpu().numpy().reshape(24,24,1)))
                    run.log({"Loss reconstruction": loss_recon.item(),
                             "Loss KLD": loss_kld.item(),
                             "Loss VAE": loss.item(),
                             "GT image": gt,
                             "Pred image": pred})
                    
                    print(
                        f'Iteration {step_idx}  '
                        f'---- reconstruction loss: {loss_recon.item():.5f} ---- | '
                        f'---- KLD loss: {loss_kld.item():.5f} ---- | '
                        f'---- VAE loss: {loss.item():.5f}')
                    
                    output_dir = os.path.join(self.saving_dir, 'vae_{}.pth'.format(timestamp))
                    
                    torch.save(self.vae.state_dict(), output_dir)

                step_idx += 1
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', '-o', default='outputs_vae', help='path to save training files', type=str)
    parser.add_argument('--beta', '-be', default=0.001, help='scale for KL divergence in VAE', type=float)
    parser.add_argument('--batch_size', '-bs', default=64, help='batch size for training', type=int)
    parser.add_argument('--num_iters', '-ni', default=10000, help='number of epochs for training', type=int)
    parser.add_argument('--device', '-d', default='cuda:0', help='device to use', type=str)
    parser.add_argument('--latent_dim', '-ldim', default=64, help='dimension of the latent space',
                        type=int)
    parser.add_argument('--num_workers', '-nw', default=4, help='number of workers for dataloader',
                        type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-2, help='learning rate for optimizer',
                        type=float)
    parser.add_argument('--unlabeled_dir', '-ud', default=None, help='path to unlabeled images', type=str)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    vae_trainer = VaeTrainer(args)
    vae_trainer.train()