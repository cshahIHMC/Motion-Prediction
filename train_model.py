"""
train_model.py

Entry point for the Motion-Prediction pipeline.
Currently: loads the dataset CSV and prints all column names.
"""

from datetime import datetime
import sys
import os
import wandb
import matplotlib.pyplot as plt
from DataLoader.data_loader import dataLoader_seq
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from Models.TCNN import TCNModel, TCNModel_Forecast
from Library import utility
import torch
import json
# Make sibling packages importable when running from the repo root.
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "Data", "trial_1_all_data.csv")
# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

## Training Function
def train_predictor_model(model, config, training_dataloader, validation_dataloader, log_wandB=False):   
    
    ## Setting up an optimizer and a loss function - Original Paper used a AdamWr optimizer We using a simple SGD
    learning_rate = config["lr"]
    momentum = config["momentum"]

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    lossFn = torch.nn.MSELoss()
    # lossFn_no_reduction = torch.nn.MSELoss(reduction='none')
    
    for batch in training_dataloader:

        Inputs, Outputs = batch
        print("Input Shape: ", Inputs.shape)
        print("Output Shape: ", Outputs.shape)
        break
    
    ## Training the periodic auto encoder
    print("Starting Training........")
    training_losses = []
    validation_losses = []

    epochs = config["epochs"]
    
    ## Training Loop
    for epoch in range(epochs):
        model.train()
    
        running_loss = 0.0
    
        for batch in training_dataloader:
        
            Inputs, Outputs = batch        
            
            Predictor_input_gpu = utility.ToDevice(Inputs)
            Predictor_input_gpu_flat = Predictor_input_gpu.reshape(Predictor_input_gpu.shape[0], -1)
            
            # 1-Time Prediction (It only works if there is a row with 1)
            Predictor_output = Outputs.squeeze(-1)  # Remove the pred_length dimension if it's 1, shape becomes [batch_size, output_features]
            
            Predictor_output_gpu = utility.ToDevice(Predictor_output)  
            
         
            # TCNN Prediction
            y_pred = model(Predictor_input_gpu)
  
                
            # Calculate the loss            
            loss = lossFn(y_pred, Predictor_output_gpu)

            # # Zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print statistics
            running_loss += loss.item() * Inputs.size(0)
                
        train_loss = running_loss / len(training_dataloader.dataset)
        training_losses.append(train_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss}')
        
        if validation_dataloader is not None:
        
            val_loss = calc_val_loss(model, validation_dataloader, lossFn)
            validation_losses.append(val_loss)

            print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}')
            
        else:
            val_loss = 0.0
        
        if log_wandB:
            wandb.log({"train/train_loss": train_loss,
                        "train/epoch": epoch,
                        "val/val_loss": val_loss,
                        "val/epoch":epoch})
        
            
    return training_losses, validation_losses  

  
# Function to calculate the PAE validation loss
def calc_val_loss(model, validation_dataloader, lossFn, lossFn_no_reduction=None):
    model.eval()
    
    val_loss = 0.0
    # individual_losses = np.zeros(6, dtype=np.float32)
    
    with torch.no_grad():
        for batch in validation_dataloader:
            
            Inputs, Outputs = batch    
            Predictor_input_gpu = utility.ToDevice(Inputs)
            Predictor_input_gpu_flat = Predictor_input_gpu.reshape(Predictor_input_gpu.shape[0], -1)
                        
            # 1-Time Prediction
            Predictor_output = Outputs.squeeze(-1)  # Remove the pred_length dimension if it's 1, shape becomes [batch_size, output_features]
            
            Predictor_output_gpu = utility.ToDevice(Predictor_output)
            

            # TCNN Prediction
            y_pred = model(Predictor_input_gpu)

            
            # Calculate the loss
            loss = lossFn(y_pred, Predictor_output_gpu)

            
            # Calculate running loss
            val_loss += loss.item() * Inputs.size(0)
            
            
        val_loss = val_loss / len(validation_dataloader.dataset)

    return val_loss

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    
    # Flags
    log_wandB = False
    train_model_flag = True
    save_file = False
    future_forcast = False
    
    prediction_horizon = 1
    
        
    if prediction_horizon != 1:
        future_forcast = True
    
    model_to_train = "TCNN" # Can be "MANN", "MoETCNN", "PAE", "RNN" 
    
    file_name = f"{model_to_train}_prediction_horizon_{prediction_horizon}"
    project_name = "Motion_Prediction April 2026"
    
        # Config the configurations
    config = {
        "training_tag": file_name,
        "project_name": project_name,
        "epochs": 30,
        "batch_size": 128,
        "num_workers": 8,
        "momentum":0.9,
        "lr": 1e-4,
        "dataset": "IHMC Senorsuit",
        "seq_length": 150,
        "pred_length": prediction_horizon,
    }
    
    ## Login to weights and biases and setup the data recording run
    if log_wandB:
        wandb.login()
        project_name = config["project_name"]
        wandb.init( project=project_name, name= config["training_tag"], config=config)
        
    # Setup Data Frames
    df = pd.read_csv(DATA_PATH)
    print("Dataset Size: ", df.shape)
    
    # Clean up the data frame - remove unwanted columns
    columns_to_extract = [ # IMU Values [7 x 6 = 42]
                         'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z', 'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z',
                         'thigh_r_acc_x', 'thigh_r_acc_y', 'thigh_r_acc_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',
                         'shank_r_acc_x', 'shank_r_acc_y', 'shank_r_acc_z', 'shank_r_gyro_x', 'shank_r_gyro_y', 'shank_r_gyro_z',
                         'R_insole_accel_x', 'R_insole_accel_y', 'R_insole_accel_z', 'R_insole_gyro_x', 'R_insole_gyro_y', 'R_insole_gyro_z',
                         'thigh_l_acc_x', 'thigh_l_acc_y', 'thigh_l_acc_z', 'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z',
                         'shank_l_acc_x', 'shank_l_acc_y', 'shank_l_acc_z', 'shank_l_gyro_x', 'shank_l_gyro_y', 'shank_l_gyro_z',
                         'L_insole_accel_x', 'L_insole_accel_y', 'L_insole_accel_z', 'L_insole_gyro_x', 'L_insole_gyro_y', 'L_insole_gyro_z',
                         
                         # Force Insole Values [6]
                         'R_insole_force', 'L_insole_force',
                         'R_insole_COPx', 'R_insole_COPz', 'L_insole_COPx', 'L_insole_COPz',
                         
                         # Filtered EMG [8]
                         'R_RF', 'R_BF', 'R_TA', 'R_GAST',
                         'L_RF', 'L_BF', 'L_TA', 'L_GAST',
                         
                         # IK [10]
                         'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
                         'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l',
                            
                         # ID [10]
                        'hip_flexion_r_moment', 'hip_adduction_r_moment', 'hip_rotation_r_moment', 'knee_angle_r_moment', 'ankle_angle_r_moment',
                        'hip_flexion_l_moment', 'hip_adduction_l_moment', 'hip_rotation_l_moment', 'knee_angle_l_moment', 'ankle_angle_l_moment']
                         
    df_modified = df[columns_to_extract]
    print("Modified Dataset Size: ", df_modified.shape)
    
    # Setup Dataset and Dataloader
    full_dataset = dataLoader_seq(df_modified, seq_length=config["seq_length"], predictor_seq_length=config["pred_length"])
    full_dataloader = DataLoader(full_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    train_dl_plot = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    
    # Setup the Model
    model = utility.ToDevice(TCNModel(
                                        input_size=48,
                                        output_size=28,
                                        num_channels=[80, 80, 80, 80, 80],
                                        kernel_size=6,
                                        dropout=0.2))
    
    # Train
    train_loss, val_loss = train_predictor_model(model=model, config=config, training_dataloader=train_dl,
                                                       validation_dataloader=val_dl, log_wandB=log_wandB)
    
    # Saving Trained Model
    if save_file:
            # Save the Model
            model_save_location = "Saved Models/"  + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"] + ".pth"
            torch.save(model.state_dict(), model_save_location)
            
            model_dict = {
                # Architecture
                "model_class": model.__class__.__name__,
                "pae_input_size": 21,
                "mann_input_size": 46,
                "output_size": 20,
                "pae_window_size": 201,
                "mann_window_size": 50,
                # "hidden_size": model.hidden_size,
                # "num_layers": model.num_layers,

                # Training hyperparameters
                "optimizer": "Adam",
                "learning_rate": config["lr"],
                "loss_fn": "MSELoss",

                # Data normalization
                "data_input_mean": full_dataset.input_mean.tolist(),
                "data_input_std": full_dataset.input_std.tolist(),
                "data_output_mean": full_dataset.output_mean.tolist(),
                "data_output_std": full_dataset.output_std.tolist()
                }


            model_save_location = "Saved Models/"  + datetime.now().strftime('%Y%m%d_%H%M') + "_" + config["training_tag"]
            with open(model_save_location + '.json', 'w') as file:
                json.dump(model_dict, file, indent=4)
                
    
    # Plotting results of the trained model  
    utility.plot_prediction(full_dataloader, model, df_modified.columns[48:76], full_dataset) 
    # utility.plot_prediction(val_dl, model, df_modified.columns[48:76], full_dataset) 


if __name__ == "__main__":
    main()
