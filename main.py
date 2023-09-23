import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
import torchvision.models as models
from torchvision.models import resnet18
from tqdm import tqdm
#from transformers import BertModel, BertConfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import torchvision.transforms
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score,average_precision_score, precision_recall_curve
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.models import resnet18
from tqdm import tqdm
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
#from utils import *
import logging
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import argparse
import configparser

from Dataload import search_data, get_sample_indices, normalization, read_and_generate_dataset, load_graphdata_channel1, get_adjacency_matrix
from Module import GCNModel


# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/METR_LA_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
data = np.load(graph_signal_matrix_filename)
data['data'].shape

all_data = read_and_generate_dataset(graph_signal_matrix_filename, 0, 0, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)

data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #torch.device('cuda:0')
print("Device:", DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)


train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, node_number = 307, var_dim = 4, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.node_number = node_number
        self.var_dim = var_dim
        self.prior = torch.distributions.MultivariateNormal(torch.zeros(var_dim).to(device), torch.eye(var_dim).to(device))

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[ : , None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[ : , None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, edge_index_info, ground_truth, c=None):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.node_number, self.var_dim)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                if c is not None:
                    input = torch.concat((c, x),-1)
                predicted_noise = model(input, edge_index_info, t)
                alpha = self.alpha[t][ : , None, None]
                alpha_hat = self.alpha_hat[t][ : , None, None]
                beta = self.beta[t][ : , None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            loss_sample = nn.MSELoss()(x, ground_truth)
            print(loss_sample.item())
            encoder_input = x.detach().cpu().numpy()
            data = encoder_input
            real = ground_truth.detach().cpu().numpy()
            data_re=real
            plt.figure(figsize=(12,6))
            fig, axs = plt.subplots(1, data.shape[-1])
            for i, axis in enumerate(axs):
                axis.hist(data[0,:,i], color='g', alpha=0.5, label="gen.")
                axis.hist(data_re[0,:, i], color='r', alpha=0.5, label="true")
            plt.legend()
            plt.show()

        model.train()
        return x, ground_truth

run_name = "try_condition"
setup_logging(run_name)
logger = SummaryWriter(os.path.join("runs", run_name))

device = torch.device("cuda")
train_envs = [4,7]
val_envs = [5.5, 6.6]
test_envs = [8]

mse = nn.MSELoss()
diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                      var_dim=4, device=device)

adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse
sparse_adjacency_matrix = scipy.sparse.csr_matrix(adj_mx)
edge_index = from_scipy_sparse_matrix(sparse_adjacency_matrix)
edge_index_info, _ = edge_index
edge_index_info = edge_index_info.to(device)

# create model
mean_train, std_train = _mean, _std
model = GCNModel(time_dim = 256, device = device, input_shape = 12,input_dim=256, hidden_dim=64, output_dim=4).to(device)

# optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,500], gamma=0.1)

l = len(train_loader)

number_of_epochs = 2000

for epoch in range(number_of_epochs):
    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(train_loader)
    for i, x in enumerate(pbar):
        encoder_inputs, labels  = x
        c = encoder_inputs[ :, :, :-1, : ].reshape(encoder_inputs.shape[0],encoder_inputs.shape[1],-1)
        x = encoder_inputs[ :, :, -1:, : ].reshape(encoder_inputs.shape[0],encoder_inputs.shape[1],-1)
        t = diffusion.sample_timesteps(x.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(x, t)

        predicted_noise = model(torch.concat((c, x_t),-1), edge_index_info, t)
        pred_std = torch.std(predicted_noise, dim=0)
        loss_std = mse(pred_std, torch.ones_like(pred_std, device=device))

        loss = mse(noise, predicted_noise) #+ 5*loss_std - 0.5*diffusion.prior.log_prob(predicted_noise.float()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(MSE=loss.item())

        logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

    scheduler.step()

    # sample
    if epoch %10 ==0:
        logging.info(f"lr = {scheduler.get_last_lr()[0]}")
        sampled_data, ground_truth = diffusion.sample(model, n=x.shape[0], edge_index_info = edge_index_info, ground_truth = x, c=c)
        #save_data(sampled_data, x, os.path.join("/content/IMAGE", run_name, f"{epoch}.jpg"))
        #torch.save(model.state_dict(), os.path.join("models_graph", run_name, f"ckpt.pt"))


