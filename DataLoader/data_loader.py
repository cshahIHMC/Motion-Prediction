import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Optional, Tuple, Dict, Iterable
from collections import defaultdict
import random

class dataLoader_seq(Dataset):
    def __init__(self, df, seq_length, predictor_seq_length=None):
        
        
        # Save all the data
        self.original_df = df.copy()
        self.seq_length = seq_length
        self.predictor_seq_length = predictor_seq_length
        
        # Store the indices
        self.indices = df.index.tolist()
        
        # # Normalize the data (column-wise)
        self.input_mean = self.original_df.iloc[:, :48].mean()
        self.input_std = self.original_df.iloc[:, :48].std()
        self.input_std[self.input_std == 0] = 1
                
        self.Input_normalized_df = (self.original_df.iloc[:, :48] - self.input_mean) / self.input_std
        
        self.output_mean = self.original_df.iloc[:, 48:76].mean()
        self.output_std = self.original_df.iloc[:, 48:76].std()
        self.output_std[self.output_std == 0] = 1
                
        self.Output_normalized_df = (self.original_df.iloc[:, 48:76] - self.output_mean) / self.output_std
        
    def __len__(self):
        return len(self.indices) - self.seq_length - self.predictor_seq_length 

    
    def __getitem__(self,idx):
        
        ######### Extract Sequences
        input_row_start_idx = self.indices[idx]
        input_row_end_idx = self.indices[idx + self.seq_length]
        
        prediction_row_idx = self.indices[idx + self.seq_length + self.predictor_seq_length]
        
        # Extract all col with that sequence length of data
        input_rows = self.Input_normalized_df.iloc[input_row_start_idx:input_row_end_idx, :].values
        output_rows = self.Output_normalized_df.iloc[prediction_row_idx, :].values
        
        # Inputs ( Transpose it to give cols, sequence length data)
        inputs = torch.tensor(input_rows , dtype=torch.float32).T
        outputs = torch.tensor(output_rows , dtype=torch.float32)
        

        # return PAE_inputs_centered, Predictor_inputs, Predictor_outputs
        return inputs, outputs
        

