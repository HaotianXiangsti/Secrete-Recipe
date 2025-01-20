import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import configparser
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse
from tensorboardX import SummaryWriter

from Dataload import (
    read_and_generate_dataset,
    load_graphdata_channel1,
    get_adjacency_matrix,
    read_and_generate_dataset_SafeGraph
)
from Module import GCNModel
from model.ASTGCN_r import make_model

from lib.utils import predict_and_save_results_mstgcn

# Import the JointTrainer class
from joint_training import JointTrainer

def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/METR_LA_astgcn.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    
    # Read config file
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    data_config = config['Data']
    training_config = config['Training']
    
    # Basic setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    # Load dataset parameters
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
    
    # Load training parameters
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    batch_size = int(training_config['batch_size'])
    
    # Load data
    if dataset_name == "NYC":
        perciption_file_path = data_config["perciption_csv"]
        data_seq, all_data, _ = read_and_generate_dataset_SafeGraph(
            graph_signal_matrix_filename,
            perciption_file_path,
            0, 0,
            num_of_hours,
            num_for_predict,
            points_per_hour=points_per_hour,
            save=True
        )
    else:
        all_data = read_and_generate_dataset(
            graph_signal_matrix_filename,
            0, 0,
            num_of_hours,
            num_for_predict,
            points_per_hour=points_per_hour,
            save=True
        )
    
    # Load model parameters
    diffusion_para = config["Diffusion Parameter"]
    model_para = config["Model Parameter"]
    
    # Prepare data loaders
    train_x_tensor, val_x_tensor, test_x_tensor, train_loader, train_target_tensor, \
    val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
        graph_signal_matrix_filename,
        num_of_hours,
        num_of_days,
        num_of_weeks,
        device,
        batch_size
    )
    
    # Create adjacency matrix
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
    sparse_adjacency_matrix = scipy.sparse.csr_matrix(adj_mx)
    edge_index = from_scipy_sparse_matrix(sparse_adjacency_matrix)
    edge_index_info, _ = edge_index
    edge_index_info = edge_index_info.to(device)
    
    # Initialize Diffusion model
    time_dim = int(model_para["time_dim"])
    input_shape = int(model_para["input_shape"])*int(len_input/4)
    input_dim = int(model_para["input_dim"])
    hidden_dim = int(model_para["hidden_dim"])
    output_dim = int(model_para["output_dim"])*int(len_input/4)
    
    diffusion_model = GCNModel(
        time_dim=time_dim,
        device=device,
        input_shape=input_shape,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    # Initialize Classification model (ASTGCN)
    nb_block = int(training_config['nb_block'])
    in_channels = int(training_config['in_channels'])
    K = int(training_config['K'])
    nb_chev_filter = int(training_config['nb_chev_filter'])
    nb_time_filter = int(training_config['nb_time_filter'])
    time_strides = num_of_hours
    
    classification_model = make_model(
        device,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        adj_mx,
        num_for_predict,
        len_input,
        num_of_vertices
    )
    
    # Initialize Diffusion
    noise_steps = int(diffusion_para["noise_steps"])
    beta_start = float(diffusion_para["beta_start"])
    beta_end = float(diffusion_para["beta_end"])
    var_dim = len_input#int(diffusion_para["var_dim"])
    number_of_nodes = int(diffusion_para["number_of_nodes"])
    
    from main import Diffusion  # Import your Diffusion class
    diffusion = Diffusion(
        noise_steps=noise_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        var_dim=var_dim,
        node_number=number_of_nodes,
        device=device
    )
    
    # Initialize Joint Trainer
    joint_trainer = JointTrainer(
        diffusion_model=diffusion_model,
        classification_model=classification_model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        edge_index_info=edge_index_info
    )
    
    best_model_path = joint_trainer.train()  # Now returns path to best model
    
    # Load the best model checkpoint
    print("Loading best model checkpoint for final evaluation...")
    joint_trainer.classification_model.load_state_dict(
        torch.load(best_model_path)
    )

    # Evaluate using the predict_and_save_results_mstgcn function
    prediction = predict_and_save_results_mstgcn(
        net=joint_trainer.classification_model,
        data_loader=joint_trainer.val_loader,
        data_target_tensor=val_target_tensor,  # This comes from your data loading
        global_step=joint_trainer.best_epoch,  # We'll need to track this
        metric_method='mask',
        _mean=_mean,
        _std=_std,
        params_path=os.path.join("results", joint_trainer.run_name),
        type='val'
    )
    
    print("Final evaluation completed!")


if __name__ == "__main__":
    main()
