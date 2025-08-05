import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob

def load_losses(model_type, alpha_pos, alpha_neg, data_name, metric):
    """Load loss metric (train/test loss or reconstruction) for given config."""
    path = f"./models_{data_name}"
    losses = []
    for seed in [1, 3, 5]:
        file_path = os.path.join(
            path, f"{model_type}_{alpha_pos}_{alpha_neg}_{data_name}_{metric}.pt")
        if os.path.exists(file_path):
            losses.append(torch.load(file_path))
        else:
            print("File not found:", file_path)
    return losses

def plot_test_loss_mnist():
    """Recreate Figure 7: Test loss over epochs for MNIST"""
    configs = [
        ('vr', 0.5, -1),
        ('vr', 2, -1),
        ('vr', 5, -1),
        ('vrs', 0.5, -0.5),
        ('vrs', 2, -2),
    ]
    labels = [
        "VR$_{0.5}$", "VR$_2$", "VR$_5$", "VRS$_{0.5,-0.5}$", "VRS$_{2,-2}$"
    ]
    linestyles = ['-', '-', '-', '--', '--']

    plt.figure(figsize=(7, 4))
    for (model_type, alpha_pos, alpha_neg), label, ls in zip(configs, labels, linestyles):
        test_losses = load_losses(model_type, alpha_pos, alpha_neg, "MNIST", "test_losses")
        if not test_losses:
            continue
        avg_len = min(len(x) for x in test_losses)
        avg_curve = np.mean([x[:avg_len] for x in test_losses], axis=0)
        plt.plot(np.linspace(0, 100, avg_len), avg_curve, label=label, linestyle=ls)

    plt.xlabel("Epoch percentage")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure7_test_loss_mnist.png")
    plt.close()

def plot_reconstruction_mse():
    """Recreate Figure 8: MSE comparison bar plot"""
    domains = ['USPS', 'MNIST', 'SVHN']
    models = [
        ('vae', 1, -1),
        ('vr', 0.5, -1),
        ('vr', 0.5, -0.5),
        ('vrs', 0.5, -0.5),
    ]
    labels = ['VAE', 'VR$_{0.5}$', 'VRLU$_{-0.5}$', 'VRS$_{0.5,-0.5}$']
    colors = ['#ffb07c', '#8e063b', '#e60049', '#f9b4ab']

    bar_width = 0.2
    x = np.arange(len(domains))
    plt.figure(figsize=(7, 4))

    for i, (model_type, alpha_pos, alpha_neg) in enumerate(models):
        means = []
        for domain in domains:
            recon = load_losses(model_type, alpha_pos, alpha_neg, domain, "test_recon_losses")
            if not recon:
                means.append(np.nan)
                continue
            mse = [x[0] for x in recon]  # extract MSE from tuple
            means.append(np.mean(mse))
        plt.bar(x + i * bar_width, means, width=bar_width, label=labels[i], color=colors[i])

    plt.xticks(x + bar_width * 1.5, domains)
    plt.ylabel("Reconstruction MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure8_mse_comparison.png")
    plt.close()

def plot_log_likelihood_pie():
    """Recreate Figure 10: Log likelihood on PIE"""
    domains = ['PIE05', 'PIE07', 'PIE09']
    models = [
        ('vae', 1, -1),
        ('vr', 0.5, -1),
        ('vr', 2, -1),
        ('vrs', 0.5, -0.5),
        ('vrs', 0.5, -2),
        ('vrs', 2, -0.5),
    ]
    labels = ['VAE', 'VR$_{0.5}$', 'VR$_2$', 'VRS$_{0.5,-0.5}$', 'VRS$_{0.5,-2}$', 'VRS$_{2,-0.5}$']
    colors = ['skyblue', 'lightgreen', 'salmon', 'mediumpurple', 'gold', 'plum']

    bar_width = 0.12
    x = np.arange(len(domains))
    plt.figure(figsize=(7, 5))

    for i, (model_type, alpha_pos, alpha_neg) in enumerate(models):
        means = []
        for domain in domains:
            log_p = load_losses(model_type, alpha_pos, alpha_neg, domain, "test_log_p_vals")
            if not log_p:
                means.append(np.nan)
                continue
            means.append(np.mean(log_p))
        plt.bar(x + i * bar_width, means, width=bar_width, label=labels[i], color=colors[i])

    plt.xticks(x + bar_width * len(models) / 2, domains)
    plt.ylabel("Marginal Log Likelihood Estimations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure10_log_likelihood_pie.png")
    plt.close()

# If you want to run it after training automatically:
if __name__ == "__main__":
    plot_test_loss_mnist()
    plot_reconstruction_mse()
    plot_log_likelihood_pie()
