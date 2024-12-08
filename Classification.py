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

from Dataload import search_data, get_sample_indices, normalization, read_and_generate_dataset, load_graphdata_channel1, get_adjacency_matrix, process_safegraph_adjmatrix, load_graphdata_channel_evaluation, read_and_generate_dataset_aug, load_graphdata_channel_aug, read_and_generate_dataset_SafeGraph, read_and_generate_dataset_SafeGraph_test
from Module import GCNModel
from model.ASTGCN_r import make_model

from lib.utils import compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from lib.metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse

from tensorboardX import SummaryWriter

from time import time
import shutil
import argparse
import configparser





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

if dataset_name == "NYC":
    perciption_file_path = data_config["perciption_csv"]
    data_seq, all_data, _ =read_and_generate_dataset_SafeGraph_test(graph_signal_matrix_filename, perciption_file_path,
                                       0,0,
                                       num_of_hours, num_for_predict,
                                       points_per_hour=points_per_hour, save=True)
    print("After it", len(all_data))
    print(data_seq.shape)

else:
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

# Check Repeated Parameters

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

train_x_tensor, val_x_tensor, test_x_tensor, train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)


adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, len_input, num_of_vertices)

print(net, flush=True)

run_name = str(training_config['run_name']) #"try_condition"

train_aug_loader_dict = {}

train_aug_loader_list = list()

train_aug_loader_list.append(train_loader)

number_envs = int(training_config['number_envs'])



for s in range(number_envs):

    aug_path = "aug/" + run_name + "/environment" + str(s) + ".npz"
    read_and_generate_dataset_aug(aug_path, 0, 0, num_of_hours, num_for_predict, points_per_hour=points_per_hour,
                                  save=True, env_number=s)
    train_loader_aug, _, _, _, _, _, _, _ = load_graphdata_channel_aug(
        aug_path, num_of_hours,
        num_of_days, num_of_weeks, DEVICE, batch_size, env_number = s)

    dataset_name = f'dataset_{s + 1}'

    train_aug_loader_dict[dataset_name] = train_loader_aug
    train_aug_loader_list.append(train_loader_aug)






l2_weights = 0.00110794568

from torch import autograd

class IRM_Calculation():
  def __init__(self, l2_weight,loss_fun,penalty_weight) -> None:
       super(IRM_Calculation, self).__init__()
       self.l2_weights=l2_weights
       self.mean_all=loss_fun #nn.functional.binary_cross_entropy_with_logits
       self.penalty_weight=penalty_weight

  def penalty(self,logits, y):

    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = self.mean_all(logits*scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]

    return torch.sum(grad**2)

  def IRM(self,logits, y,model):

    weight_norm = torch.tensor(0.).to(device)
    for w in model.parameters():
      weight_norm += w.norm().pow(2)
    loss=self.mean_all(logits, y).clone()
    loss += self.l2_weights * weight_norm
    loss += self.penalty_weight * self.penalty(logits, y)

    return loss

def aug_train(encoder_inputs,labels_aug,missing_value,masked_flag,criterion):

    criterion_masked = masked_mae
    outputs_aug = net(encoder_inputs)
    if masked_flag:
        loss_aug = criterion_masked(outputs_aug, labels_aug, missing_value)
    else:
        loss_aug = criterion(outputs_aug, labels_aug)

    return outputs_aug, loss_aug

def merge_train_datasets(original_loader, aug_loader_list):
    """
    Merge multiple data loaders into a single dataset
    
    Args:
        original_loader: Original training data loader
        aug_loader_list: List of augmented data loaders
    
    Returns:
        DataLoader: Combined data loader with all datasets
    """
    # Get all datasets from the loaders
    all_datasets = [original_loader.dataset]
    for aug_loader in aug_loader_list:
        all_datasets.append(aug_loader.dataset)
    
    # Combine all datasets using ConcatDataset
    combined_dataset = ConcatDataset(all_datasets)
    
    # Create new loader with combined dataset using same batch size as original
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=original_loader.batch_size,
        shuffle=True,
        num_workers=original_loader.num_workers if hasattr(original_loader, 'num_workers') else 0
    )
    
    return combined_loader

def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag=0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function=='masked_mse':
        criterion_masked = masked_mse         #nn.MSELoss().to(DEVICE)
        masked_flag=1
    elif loss_function=='masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag= 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag,missing_value,sw, epoch)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)


        if val_loss < best_val_loss:

            files = os.listdir(params_path)

            # 遍历文件夹中的文件
            for filename in files:
                if filename.endswith('.params'):
                    # 构建文件的完整路径
                    file_path = os.path.join(params_path, filename)

                    # 删除文件
                    os.remove(file_path)

            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode

        #for batch_index, batch_data in enumerate(train_loader):

        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data

            optimizer.zero_grad()

            outputs = net(encoder_inputs)

            if masked_flag:
                loss = criterion_masked(outputs, labels,missing_value)
            else :
                loss = criterion(outputs, labels)

            aug = False
            irm_calculation = IRM_Calculation(l2_weights, criterion, 1)
            # loss=irm_calculation.IRM(outputs,labels,net)

            if aug:

                loss_aug_list =list()

                """
                data_batch_aug_list = [batch_data_1, batch_data_2, batch_data_3, batch_data_4, batch_data_5, batch_data_6]

                for batch_data_aug in data_batch_aug_list:
                    encoder_inputs_aug, labels_aug = batch_data_aug
                    outputs_aug, loss_aug = aug_train(encoder_inputs_aug, labels_aug, missing_value, masked_flag,
                                                      criterion)
                    loss_aug = irm_calculation.IRM(outputs_aug, labels_aug, net)

                    loss_aug_list.append(loss_aug)

                """

                
                for s in range(len(train_aug_loader_dict)):

                    dataset_name = f'dataset_{s + 1}'

                    train_loader_aug = train_aug_loader_dict[dataset_name]

                    for batch_index_aug, batch_data_aug in enumerate(train_loader_aug):

                        encoder_inputs_aug, labels_aug = batch_data_aug
                        outputs_aug, loss_aug = aug_train(encoder_inputs_aug, labels_aug, missing_value,masked_flag, criterion)
                        loss_aug = irm_calculation.IRM(outputs_aug, labels_aug, net)
                        loss_aug_list.append(loss_aug)
                

                loss = loss + sum(loss_aug_list)




            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 1000 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

    print('best epoch:', best_epoch)

    prediction = predict_main(best_epoch, test_loader, test_target_tensor, metric_method, _mean, _std, 'test')

    # apply the best model on the test set
    '''
    prediction = predict_main(best_epoch, test_loader, test_target_tensor,metric_method ,_mean, _std, 'test')

    #total_data = torch.concat((train_target_tensor, val_target_tensor, test_target_tensor), dim = 0)

    #total_data = torch.concat((train_target_tensor, val_target_tensor, test_target_tensor), dim=0)




    total_data = torch.concat((train_target_tensor, val_target_tensor, test_target_tensor), dim=0).detach().cpu().numpy() * 9432.49867276 + 5981.70799134
    total_data = np.concatenate((data_seq[:9,:, :], total_data), axis = 0)
    prediction = np.concatenate((data_seq[:9,:, ], train_target_tensor[:,:,:].detach().cpu().numpy(), val_target_tensor[:,:,:].detach().cpu().numpy(), prediction), axis = 0)*9432.49867276+5981.70799134
    np.save("prediction.npy", prediction)
    #total_data = total_data.detach().cpu().numpy() * 9432.49867276 + 5981.70799134
    np.save("total_data.npy", total_data)

    x = np.arange(total_data.shape[0])
    plt.plot(x[-18:], prediction[:,1,-1][-18:], linestyle='--', linewidth=3, label = "37527 Prediction" )
    plt.plot(x[-18:], prediction[:, 8, -1][-18:], linestyle='--', linewidth=3, label = "14307 Predition")


    plt.plot(x, total_data[:,1,-1], label = "37527 Observation")
    plt.plot(x, total_data[:, 8, -1], label = "14307 Observation")

    plt.xlim(1,90)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(data_seq[:,:,-1])
    plt.show()
    '''



    ''''
    plt.figure(figsize=(8, 7))
    prediction = np.concatenate(
        (train_target_tensor.detach().cpu().numpy(), val_target_tensor.detach().cpu().numpy(), prediction), axis=0)
    total_data = torch.concat((train_target_tensor, val_target_tensor, test_target_tensor), dim=0)
    
    # 创建 x 轴的刻度
    x_ticks = np.arange(0, 1440, 180)  # 每隔 60 个点（每隔 5 小时）设置一个刻度

    # 创建 x 轴的标签
    x_labels = [f"{hour:02d}:00" for hour in range(0, 24, 3)]  # 设置每隔 5 小时显示一个小时标签

    # 绘制数据
    plt.plot(prediction[:, 35, -1][-1440:], linestyle='--', linewidth=3, label="Sensor NO.35 Prediction")
    #plt.plot(prediction[:, 57, -1][-1440:], linestyle='--', linewidth=3, label="Sensor NO.57 Prediction")

    print(prediction[1, 35, -1])


    total_data = total_data.detach().cpu().numpy()

    np.save("total_data.npy",total_data)

    plt.plot(total_data[:, 35, -1][-1440:], label="Sensor NO.35 Observation")
    #plt.plot(total_data[:, 57, -1][-1440:], label="Sensor NO.57 Observation")

    # 设置 x 轴的刻度和标签
    plt.xticks(x_ticks, x_labels, rotation=45)  # 旋转标签以便显示

    # 添加标签和标题
    plt.xlabel('Time')
    plt.ylabel('Flow')
    plt.legend()

    # 显示图形


    plt.savefig('my_plot.png')
    plt.show()
    '''





def predict_main(global_step, data_loader, data_target_tensor,metric_method, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)


    net.load_state_dict(torch.load(params_filename))

    prediction = predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method,_mean, _std, params_path, type)

    return prediction


if __name__ == "__main__":

    train_main()
