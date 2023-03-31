import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial
import matplotlib.pyplot as plt
import data as Data
import datetime
import os
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)


def elbo(model, x, z, mu, logstd, gamma=1):
    # decoded
    x_hat = model.decode(z)

    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = 0.5 * torch.sum(torch.exp(logstd) - logstd - 1 + mu.pow(2))
    # KLD = 0.5 * (torch.exp(logstd) - logstd - 1 + mu.pow(2))
    loss = BCE + gamma * KLD

    return loss


# Variational Renyi

def renyi_bound(method, model, x, z, mu, logstd, alpha, K, testing_mode=False):
    #print("mu = ", mu)
    #print("z = ", z)
    log_q = model.compute_log_probabitility_gaussian(z, mu, logstd)

    log_p_z = model.compute_log_probabitility_gaussian(z, torch.zeros_like(z, requires_grad=False),
                                                       torch.zeros_like(z, requires_grad=False))

    x_hat = model.decode(z)

    log_p = model.compute_log_probabitility_bernoulli(x_hat, x)
    # log_p = model.compute_log_probabitility_gaussian(x_hat, x, torch.zeros_like(x_hat))

    log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1 - alpha)

    loss = 0
    if alpha == 1:
        loss = elbo(model, x, z, mu, logstd)
    if method == 'vr':
        loss = compute_MC_approximation(log_w_matrix, alpha, testing_mode)
    elif method == 'vr_ub':
        loss = compute_approximation_for_negative_alpha(log_w_matrix, alpha)
    else:
        print("Invalid value of alpha")

    return loss


def compute_MC_approximation(log_w_matrix, alpha, testing_mode=False):
    log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
    ws_matrix = torch.exp(log_w_minus_max)
    ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

    if not testing_mode:
        sample_dist = Multinomial(1, ws_norm)
        ws_sum_per_datapoint = log_w_matrix.gather(1, sample_dist.sample().argmax(1, keepdim=True))
    else:
        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)

    if alpha == 1:
        print("Invalid value of alpha")
        return

    ws_sum_per_datapoint /= (1 - alpha)

    loss = -torch.sum(ws_sum_per_datapoint)
    # loss = -ws_sum_per_datapoint
    return loss


def compute_approximation_for_negative_alpha(log_w_matrix, alpha):
    norm_log_w_matrix = log_w_matrix.view(log_w_matrix.size(0), -1)

    min_val = norm_log_w_matrix.min(1, keepdim=True)[0]
    max_val = norm_log_w_matrix.max(1, keepdim=True)[0]

    norm_log_w_matrix -= min_val
    norm_log_w_matrix /= max_val
    norm_w_matrix = torch.exp(norm_log_w_matrix)

    approx = norm_w_matrix - 1
    approx *= max_val
    approx += min_val

    ws_norm = approx / torch.sum(approx, 1, keepdim=True)
    ws_sum_per_datapoint = torch.sum(approx * ws_norm, 1)

    ws_sum_per_datapoint /= (1 - alpha)

    loss = -torch.sum(ws_sum_per_datapoint)
    # loss = -ws_sum_per_datapoint
    return loss


# Variational Renyi - average approximation for positive and negative α
def renyi_bound_sandwich(model, x, z, mu, logstd, alpha_pos, alpha_neg, K, testing_mode=False):
    loss_pos = renyi_bound('vr', model, x, z, mu, logstd, alpha_pos, K, testing_mode)
    loss_neg = renyi_bound('vr_ub', model, x, z, mu, logstd, alpha_neg, K, testing_mode)

    loss = (loss_neg + loss_pos) / 2
    return loss



# Model

class vr_model(nn.Module):
    def __init__(self, alpha_pos, alpha_neg):
        super(vr_model, self).__init__()

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)
        self.fc32 = nn.Linear(200, 50)

        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        h3 = torch.tanh(self.fc4(z))
        h4 = torch.tanh(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(nn.Parameter(torch.Tensor([0.0])))
        scale = scale.to(device)
        mean = x_hat

        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum()

    def MSE_reconstruction_error(self, x_hat, x):
        return torch.sum(torch.mean(torch.pow(x - x_hat, 2), axis=1))

    def CE_reconstruction_error(self, x_hat, x):
        loss = -torch.sum(x * torch.log(x_hat))
        return loss / x_hat.size(dim=0)

    def compute_log_probabitility_gaussian(self, obs, mu, logstd, axis=1):
        std = torch.exp(logstd)
        n = Normal(mu, std)
        res = torch.mean(n.log_prob(obs), axis)
        return res

    def compute_log_probabitility_bernoulli(self, obs, p, axis=1):
        return torch.sum(p * torch.log(obs) + (1 - p) * torch.log(1 - obs), axis)

    def compute_loss_for_batch(self, data, model, model_type, K, testing_mode=False):
        # data = (B, 1, H, W)
        B, _, H, W = data.shape
        x = data.repeat((1, K, 1, 1)).view(-1, H * W)
        mu, logstd = model.encode(x)
        # (B*K, #latents)
        z = model.reparameterize(mu, logstd)
        loss = 0
        if model_type == "vae":
            loss = elbo(model, x, z, mu, logstd)
        elif model_type == "vr":
            loss = renyi_bound("vr", model, x, z, mu, logstd, model.alpha_pos, K, testing_mode)
        elif model_type == "vrs":
            loss = renyi_bound_sandwich(model, x, z, mu, logstd, model.alpha_pos, model.alpha_neg, K, testing_mode)

        # reconstruction loss
        x_hat = model.decode(z)
        recon_loss_MSE = model.MSE_reconstruction_error(x_hat, x)
        recon_loss_CE = model.CE_reconstruction_error(x_hat, x)
        log_p = model.compute_log_probabitility_bernoulli(x_hat, x)
        tmp1 = log_p.view(B, K)
        tmp2 = torch.mean(tmp1, 1)
        return loss, recon_loss_MSE, recon_loss_CE, torch.sum(tmp2)



# Train and Test

def train(model, optimizer, epoch, train_loader, model_type, losses, recon_losses, log_p_vals):
    model.train()
    train_loss = 0
    train_recon_loss_MSE = 0
    train_recon_loss_CE = 0
    train_log_p = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # print("batch_idx = ", batch_idx)
        data = data.to(device)
        optimizer.zero_grad()

        loss, recon_loss_MSE, recon_loss_CE, log_p = model.compute_loss_for_batch(data, model, model_type, K=50)
        loss.backward()
        train_loss += loss.item()
        train_recon_loss_MSE += recon_loss_MSE.item()
        train_recon_loss_CE += recon_loss_CE.item()
        train_log_p += log_p.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    losses.append(train_loss / len(train_loader.dataset))
    recon_losses.append(
        (train_recon_loss_MSE / len(train_loader.dataset), train_recon_loss_CE / len(train_loader.dataset)))
    log_p_vals.append(train_log_p / len(train_loader.dataset))
    return losses, recon_losses, log_p_vals


def test(model, epoch, test_loader, model_type, losses, recon_losses, log_p_vals):
    model.eval()
    test_loss = 0
    test_recon_loss_MSE = 0
    test_recon_loss_CE = 0
    test_log_p = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss_MSE, recon_loss_CE, log_p = model.compute_loss_for_batch(data, model, model_type, K=50,
                                                                                      testing_mode=True)
            test_loss += loss.item()
            test_recon_loss_MSE += recon_loss_MSE.item()
            test_recon_loss_CE += recon_loss_CE.item()
            test_log_p += log_p.item()
            if i == 0:
                # plt.style.use('classic')
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.reshape(recon_batch.shape[0], 1, 28, 28)[:n]])

                f, axarr = plt.subplots(2, 8)
                for j in range(8):
                    axarr[0, j].imshow(comparison.cpu()[j, 0], interpolation='nearest', cmap='viridis')
                for j in range(8, 16):
                    axarr[1, j - 8].imshow(comparison.cpu()[j, 0], interpolation='nearest', cmap='viridis')
                os.makedirs('reconstructed_images', exist_ok=True)
                plt.savefig('reconstructed_images/{}.png'.format(epoch))

    test_loss /= len(test_loader.dataset)
    test_recon_loss_MSE /= len(test_loader.dataset)
    test_recon_loss_CE /= len(test_loader.dataset)
    test_log_p /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}, alpha_pos: {:.4f}, alpha_neg: {:.4f}'.format(
        test_loss, model.alpha_pos, model.alpha_neg))
    losses.append(test_loss)
    recon_losses.append((test_recon_loss_MSE, test_recon_loss_CE))
    log_p_vals.append(test_log_p)
    return losses, recon_losses, log_p_vals


# Run

def run(model_type, alpha_pos, alpha_neg, data_name, seed):
    learning_rate = 0.001
    testing_frequency = 50

    train_losses, test_losses = [], []
    train_recon_losses, test_recon_losses = [], []
    train_log_p_vals, test_log_p_vals = [], []

    model = vr_model(alpha_pos, alpha_neg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, test_loader = Data.get_data_loaders(data_name, seed)


    print(datetime.datetime.now())
    eps = 1e-4
    epoch = 0
    testing_cnt = 0
    while True:
        if epoch > 300:
            break

        # stop learning condition
        if testing_cnt >= 3 and test_losses[-1] >= test_losses[-2] and test_losses[-2] >= test_losses[-3]:
            break

        # stop learning condition
        if len(train_losses) >= 3 and np.abs(train_losses[-1] - train_losses[-2]) <= eps and \
                np.abs(train_losses[-2] - train_losses[-3]) <= eps:
            break

        train_losses, train_recon_losses, train_log_p_vals = train(model, optimizer, epoch, train_loader,
                                                                   model_type, train_losses, train_recon_losses,
                                                                   train_log_p_vals)
        if epoch % testing_frequency == 1:
            test_losses, test_recon_losses, test_log_p_vals = test(model, epoch, test_loader,
                                                                   model_type, test_losses, test_recon_losses,
                                                                   test_log_p_vals)
            testing_cnt += 1

        epoch += 1

    print(datetime.datetime.now())
    print("Training finished")

    path = "./models_{}".format(data_name, seed)
    os.makedirs(path, exist_ok=True)

    torch.save(train_losses,
               path + "/{}_{}_{}_{}_train_losses.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(train_recon_losses,
               path + "/{}_{}_{}_{}_train_recon_losses.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(train_log_p_vals,
               path + "/{}_{}_{}_{}_train_log_p_vals.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(test_losses,
               path + "/{}_{}_{}_{}_test_losses.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(test_recon_losses,
               path + "/{}_{}_{}_{}_test_recon_losses.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(test_log_p_vals,
               path + "/{}_{}_{}_{}_test_log_p_vals.pt".format(model_type, alpha_pos, alpha_neg, data_name))
    torch.save(model.state_dict(), path + "/{}_{}_{}_{}_model.pt".format(model_type, alpha_pos, alpha_neg, data_name))


def main():
    domains = ['MNIST', 'USPS', 'SVHN']

    for domain in domains:
        for seed in [1, 3, 5]:
            torch.manual_seed(seed)
            run('vae', alpha_pos=1, alpha_neg=-1, data_name=domain, seed=seed)
            run('vr', alpha_pos=2, alpha_neg=-1, data_name=domain, seed=seed)
            run('vr', alpha_pos=0.5, alpha_neg=-1, data_name=domain, seed=seed)
            run('vrs', alpha_pos=2, alpha_neg=-0.5, data_name=domain, seed=seed)
            run('vrs', alpha_pos=0.5, alpha_neg=-2, data_name=domain, seed=seed)
            run('vrs', alpha_pos=0.5, alpha_neg=-0.5, data_name=domain,  seed=seed)
            run('vrs', alpha_pos=2, alpha_neg=-2, data_name=domain, seed=seed)

if __name__ == "__main__":
    main()