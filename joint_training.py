import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import os
import numpy as np
from time import time

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
        self.batch_size = int(config['Training']['batch_size'])
        self.learning_rate = float(config['Training']['learning_rate'])
        
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
        
        # Logging
        self.run_name = str(config['Training']['run_name'])
        self.setup_logging()
        self.logger = SummaryWriter(os.path.join("runs", self.run_name))
        
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

    def train_classification_step(self, encoder_inputs, labels):
        """Single training step for classification model"""
        self.classification_optimizer.zero_grad()
        outputs = self.classification_model(encoder_inputs)
        loss = self.criterion_masked(outputs, labels, 0.0)
        return loss, outputs

    def generate_augmented_data(self, encoder_inputs):
        """Generate augmented data using diffusion model"""
        with torch.no_grad():
            c = encoder_inputs[:, :, :-1, :].reshape(encoder_inputs.shape[0], encoder_inputs.shape[1], -1)
            x = encoder_inputs[:, :, -1:, :].reshape(encoder_inputs.shape[0], encoder_inputs.shape[1], -1)

            sampled_data, _, _ = self.diffusion.sample(
                self.diffusion_model,
                n=x.shape[0],
                edge_index_info=self.edge_index_info,
                ground_truth=x,
                path=os.path.join("results", self.run_name, "aug.jpg"),
                c=c
            )

            # Reshape sampled data to match original format
            sampled_data = sampled_data.unsqueeze(-2)
            c = c.reshape(c.shape[0], c.shape[1], -1, 4)  # Adjust the last dimension based on your data
            augmented_data = torch.cat((c, sampled_data), dim=-2)

            return augmented_data

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
            
            for batch_idx, (encoder_inputs, labels) in enumerate(pbar):
                loss = self.train_diffusion_step(encoder_inputs, labels)
                loss.backward()
                self.diffusion_optimizer.step()
                
                pbar.set_description(f"Diffusion Pre-training Epoch {epoch}, Loss: {loss.item():.4f}")
                self.logger.add_scalar('diffusion_pretrain_loss', loss.item(), global_step)
                global_step += 1
        
        # Phase 2: Joint training
        print("Starting joint training phase...")
        for epoch in range(self.diffusion_pretraining_epochs, self.epochs):


            self.diffusion_model.train()
            self.classification_model.train()

            pbar = tqdm(self.train_loader)

            
            for batch_idx, (encoder_inputs, labels) in enumerate(pbar):
                # Generate augmented data

                augmented_data = self.generate_augmented_data(encoder_inputs)

                
                # Train classification model
                class_loss, outputs = self.train_classification_step(encoder_inputs, labels)

                class_loss.backward()
                self.classification_optimizer.step()
                
                
                # Train diffusion model with both losses
                diff_loss = self.train_diffusion_step(encoder_inputs, labels)
                aug_class_loss, _ = self.train_classification_step(augmented_data, labels)

                
                
                total_loss = diff_loss + aug_class_loss
                total_loss.backward()
                self.diffusion_optimizer.step()
                
                pbar.set_description(
                    f"Joint Training Epoch {epoch}, "
                    f"Class Loss: {class_loss.item():.4f}, "
                    f"Diff Loss: {diff_loss.item():.4f}"
                )
                
                self.logger.add_scalar('classification_loss', class_loss.item(), global_step)
                self.logger.add_scalar('diffusion_loss', diff_loss.item(), global_step)
                global_step += 1
            
            # Validation
            val_loss = self.validate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_models(epoch)
            
            print(f"Epoch {epoch} completed in {time() - start_time:.2f}s")

            del augmented_data
            torch.cuda.empty_cache()
            
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