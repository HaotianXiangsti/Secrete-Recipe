import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import os
import numpy as np
from time import time

l2_weights = 0.00110794568

from torch import autograd

class IRM_Calculation():
  def __init__(self, device, l2_weight,loss_fun,penalty_weight) -> None:
       super(IRM_Calculation, self).__init__()
       self.device = device
       self.l2_weights=l2_weights
       self.mean_all=loss_fun #nn.functional.binary_cross_entropy_with_logits
       self.penalty_weight=penalty_weight

  def penalty(self,logits, y):

    scale = torch.tensor(1.).to(self.device).requires_grad_()
    loss = self.mean_all(logits*scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]

    return torch.sum(grad**2)

  def IRM(self,logits, y,model):

    weight_norm = torch.tensor(0.).to(self.device)
    for w in model.parameters():
      weight_norm += w.norm().pow(2)
    loss=self.mean_all(logits, y).clone()
    loss += self.l2_weights * weight_norm
    loss += self.penalty_weight * self.penalty(logits, y)

    return loss

class JointTrainer:
    def __init__(self, 
                 diffusion_model,
                 classification_model,
                 diffusion,
                 train_loader,
                 val_loader,
                 test_loader,
                 config,
                 device,
                 edge_index_info):
        """
        Initialize the joint trainer for diffusion and classification models
        """
        self.diffusion_model = diffusion_model
        self.classification_model = classification_model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.edge_index_info = edge_index_info
        
        # Training parameters
        self.epochs = int(config['Training']['epochs'])
        self.diffusion_pretraining_epochs = int(config['Training']['diff_epochs'])
        self.epochs = int(config['Training']['epochs'])+int(config['Training']['diff_epochs'])
        self.batch_size = int(config['Training']['batch_size'])
        self.learning_rate = float(config['Training']['learning_rate'])

        self.input_len = int(config['Data']['len_input'])
        
        # Setup optimizers
        self.diffusion_optimizer = optim.AdamW(
            self.diffusion_model.parameters(), 
            lr=self.learning_rate
        )
        self.classification_optimizer = optim.AdamW(
            self.classification_model.parameters(), 
            lr=self.learning_rate
        )

        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.criterion_masked = self.masked_mae

        # IRM setup
        self.l2_weights = 0.00110794568
        self.penalty_weight = 1.0  # Adjust as needed
        self.irm_calc = IRM_Calculation(
            device = self.device,
            l2_weight=self.l2_weights,
            loss_fun=self.criterion_masked,
            penalty_weight=self.penalty_weight
        )
        
        # Logging
        self.run_name = str(config['Training']['run_name'])
        self.setup_logging()
        self.logger = SummaryWriter(os.path.join("runs", self.run_name))

        self.best_epoch = 0  # Track epoch with best validation loss
        self.best_model_path = None  # Track path of best model checkpoint
        
    def setup_logging(self):
        """Set up logging directories"""
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs(os.path.join("models", self.run_name), exist_ok=True)
        os.makedirs(os.path.join("results", self.run_name), exist_ok=True)

    def masked_mae(self, preds, labels, null_val=0.0):
        """Masked MAE loss"""
        masks = ~torch.isclose(labels, torch.tensor(null_val).to(labels.device))
        masks = masks.float()
        mask_sum = torch.sum(masks)
        diff = torch.abs(preds - labels)
        mae = diff * masks
        mae = torch.sum(mae)
        mae = mae / (mask_sum + 1e-6)
        return mae

    def train_diffusion_step(self, encoder_inputs, labels):
        """Single training step for diffusion model"""
        self.diffusion_optimizer.zero_grad()
        
        c = encoder_inputs[:, :, :-1, :].reshape(encoder_inputs.shape[0], encoder_inputs.shape[1], -1)
        x = encoder_inputs[:, :, -1:, :].reshape(encoder_inputs.shape[0], encoder_inputs.shape[1], -1)
        
        t = self.diffusion.sample_timesteps(x.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(x, t)

        predicted_noise = self.diffusion_model(
            torch.concat((c, x_t), -1), 
            self.edge_index_info, 
            t
        )
        
        loss = self.mse_loss(noise, predicted_noise)
        return loss

    def train_classification_step(self, encoder_inputs, labels, irm_use = False):
        """Single training step for classification model with IRM"""
        self.classification_optimizer.zero_grad()
        outputs = self.classification_model(encoder_inputs)
        if irm_use:
            # Use IRM loss instead of standard loss
            loss = self.irm_calc.IRM(outputs, labels, self.classification_model)

        else:
            loss = self.criterion_masked(outputs, labels, 0.0)

        return loss, outputs

    def generate_augmented_data(self, encoder_inputs):
        """
        Generate augmented data and corresponding targets from diffusion model
        
        Parameters:
        -----------
        encoder_inputs: tensor of shape (batch_size, num_vertices, num_features, time_steps)
        
        Returns:
        --------
        augmented_data: tensor of shape (batch_size, num_vertices, num_features, time_steps)
        augmented_targets: tensor of shape (batch_size, num_vertices, prediction_horizon)
        """
        with torch.no_grad():
            c = encoder_inputs[:, :, :-1, :].reshape(encoder_inputs.shape[0], encoder_inputs.shape[1], -1)
            x = encoder_inputs[:, :, -1:, :].reshape(encoder_inputs.shape[0], encoder_inputs.shape[1], -1)
            
            # Generate random scaling factors between 0.5 and 2.0
            random_scales = torch.rand_like(c[:,:,-1:]) * 1.5 + 0.5
            
            # Apply random scaling to conditions
            c[:,:,-1:] = c[:,:,-1:] * random_scales
            
            sampled_data, _, _ = self.diffusion.sample(
                self.diffusion_model,
                n=x.shape[0],
                edge_index_info=self.edge_index_info,
                ground_truth=x,
                path=os.path.join("results", self.run_name, "aug.jpg"),
                c=c
            )
            
            # Reshape sampled data to match original format
            sampled_data = sampled_data.unsqueeze(-2)  # Shape: (b, v, 1, T)
            c = c.reshape(c.shape[0], c.shape[1], -1, self.input_len)  # Shape: (b, v, num_features-1, T)
            augmented_data = torch.cat((c, sampled_data), dim=-2)  # Shape: (b, v, num_features, T)
            
            # Extract target from augmented data following the original code's approach
            # Target is taken from the last feature channel (traffic flow)
            augmented_targets = augmented_data[:, :, -1, :]  # Shape: (b, v, T)
            
            return augmented_data, augmented_targets

    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        start_time = time()
        global_step = 0
        
        # Phase 1: Diffusion pre-training 
        print("Starting diffusion pre-training phase...")
        for epoch in range(self.diffusion_pretraining_epochs):
            self.diffusion_model.train()
            pbar = tqdm(self.train_loader)
            epoch_diff_loss = 0
            
            for batch_idx, (encoder_inputs, labels) in enumerate(pbar):
                loss = self.train_diffusion_step(encoder_inputs, labels)
                loss.backward()
                self.diffusion_optimizer.step()
                
                epoch_diff_loss += loss.item()
                pbar.set_description(f"Diffusion Pre-training Epoch {epoch}, Loss: {loss.item():.4f}")
                self.logger.add_scalar('diffusion_pretrain_loss', loss.item(), global_step)
                global_step += 1
                
            avg_diff_loss = epoch_diff_loss / len(self.train_loader)
            print(f"Epoch {epoch} average diffusion loss: {avg_diff_loss:.4f}")
        
        # Phase 2: Joint training
        print("Starting joint training phase...")
        for epoch in range(self.diffusion_pretraining_epochs, self.epochs):
            self.diffusion_model.train()
            self.classification_model.train()
            
            pbar = tqdm(self.train_loader)
            epoch_cls_loss = 0
            
            for batch_idx, (encoder_inputs, labels) in enumerate(pbar):
                # 清空所有梯度
                self.diffusion_optimizer.zero_grad()
                self.classification_optimizer.zero_grad()

                # 计算生成loss
                diff_loss = self.train_diffusion_step(encoder_inputs, labels)

                diff_loss.backward()

                self.diffusion_optimizer.step()

                self.diffusion_optimizer.zero_grad()
                
                # Generate augmented data
                augmented_data, augmented_targets = self.generate_augmented_data(encoder_inputs)
                
                # Concatenate original and augmented data
                combined_data = torch.cat([encoder_inputs, augmented_data], dim=0)
                combined_labels = torch.cat([labels, augmented_targets], dim=0)
                
                # 计算分类loss
                class_loss, outputs = self.train_classification_step(combined_data, combined_labels)
                
                
                class_loss.backward()

                self.diffusion_optimizer.step()

                # 更新两个优化器
                self.classification_optimizer.step()

                # 计算总loss并回传
                total_loss = class_loss + diff_loss
                
                epoch_cls_loss += class_loss.item()
                
                pbar.set_description(
                    f"Joint Training Epoch {epoch}, "
                    f"Class Loss: {class_loss.item():.4f}, "
                    f"Diff Loss: {diff_loss.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}"
                )
                
                self.logger.add_scalar('classification_loss', class_loss.item(), global_step)
                self.logger.add_scalar('diffusion_loss', diff_loss.item(), global_step)
                self.logger.add_scalar('total_loss', total_loss.item(), global_step)
                global_step += 1
            
            # Calculate and print average loss for this epoch
            avg_epoch_loss = epoch_cls_loss / len(self.train_loader)
            print(f"Epoch {epoch} average total loss: {avg_epoch_loss:.4f}")
            
            # Validation
            val_loss = self.validate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_epoch = epoch
                # Save both models but track classification model path
                self.best_model_path = os.path.join(
                    "models", 
                    self.run_name, 
                    f"classification_epoch_{epoch}.pt"
                )
                self.save_models(epoch)
            print(f"Val Loss {val_loss}")
            
            print(f"Epoch {epoch} completed in {time() - start_time:.2f}s")
            
            del augmented_data
            torch.cuda.empty_cache()
            
            return self.best_model_path  # Return path to best model checkpoint
            
    def validate(self):
        """Validation step"""
        self.classification_model.eval()
        val_loss = 0
        with torch.no_grad():
            for encoder_inputs, labels in self.val_loader:
                outputs = self.classification_model(encoder_inputs)
                val_loss += self.criterion_masked(outputs, labels, 0.0).item()
        return val_loss / len(self.val_loader)
    
    def save_models(self, epoch):
        """Save model checkpoints"""
        torch.save(
            self.diffusion_model.state_dict(),
            os.path.join("models", self.run_name, f"diffusion_epoch_{epoch}.pt")
        )
        torch.save(
            self.classification_model.state_dict(),
            os.path.join("models", self.run_name, f"classification_epoch_{epoch}.pt")
        )
