import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import missingno as msno
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import joblib
import time, math
import gc, os, multiprocessing
import copy

from functools import reduce
from operator import mul

from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

import optuna
from optuna.integration import SkorchPruningCallback

import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR


# --------------------------------------- Check CUDA --------------------------------------- #
def cuda_check(verbose=0):
    # --- Device Configuration ---
    # This determines if CUDA is available and sets the device accordingly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device) # This sets the default device for new tensor creation
    
    if verbose != 0:
        print(f"Using device: {device}")

        # Check if CUDA is available
        if torch.cuda.is_available():
            # Get the number of available GPUs
            num_gpus = torch.cuda.device_count()
            print(f"CUDA is available. Number of GPUs found: {num_gpus}")

            # Print information for each GPU
            for i in range(num_gpus):
                print(f"\n--- GPU {i} ---")
                print(f"  Device Name: {torch.cuda.get_device_name(i)}")
                # Get memory information (total and currently used)
                total_memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  Total Memory: {total_memory_gb:.2f} GB")
                
                # This will show memory currently allocated by PyTorch,
                # but not necessarily all memory reserved by the driver.
                # It's useful for checking after a training run.
                allocated_memory_mb = torch.cuda.memory_allocated(i) / 1024**2
                cached_memory_mb = torch.cuda.memory_reserved(i) / 1024**2
                print(f"  Allocated Memory: {allocated_memory_mb:.2f} MB")
                print(f"  Cached Memory: {cached_memory_mb:.2f} MB")
                
        else:
            print("CUDA is not available.")

        # Get the number of CPU cores
        num_cpu_cores = os.cpu_count()
        # or
        # num_cpu_cores = multiprocessing.cpu_count()

        print('\n---Parallel computing unit count---')
        print(f"n_jobs=-1 is equivalent to n_jobs = {num_cpu_cores}")

    return device 


# # --------------------------------------- Data Preprocessing --------------------------------------- #

def df_transform_lag_regressor(df, lag_regressor_cols, lookback_timestep):
    """
    Transforms a time-series DataFrame into input arrays suitable for
    sequence-to-sequence models (like Transformers) using a sliding window approach.

    Args:
        df (pd.DataFrame): The input DataFrame containing time-series data,
                           with a DatetimeIndex.
        lag_regressor_cols (list): A list of column names from `df` that should
                                   be used as features for the input (lagged values)
                                   and output (future values).
        lookback_timedelta (pd.Timedelta): The duration for the look-back window.
                                           e.g., pd.Timedelta(hours=1) for 1 hour lookback.
        
    Returns:
        tuple:
            - input_array (np.ndarray): Shape (num_samples, lookback_timestep, num_lag_regressors).
                                        Contains lagged features for each sample.
            - datetime (pd.DatetimeIndex): The datetime index corresponding to the
                                           start of each valid sample (after NaN removal).
    """

    input_array = []
    for i in range(lookback_timestep):
        shifted_col_numpy = df[lag_regressor_cols].shift(i+1).to_numpy()
        input_array.append(shifted_col_numpy)
    input_array = np.array(input_array).transpose(1, 0, 2)              # input_array has a shape (num_sample, lookback_timestep, num_lag_regressor)

    datetime = df.index

    return input_array, datetime

def df_transform_target(df, target_col, lookforward_timestep):
    """
    Transforms a time-series DataFrame into output arrays suitable for
    the models using a sliding window approach.

    Args:
        df (pd.DataFrame): The input DataFrame containing time-series data,
                           with a DatetimeIndex.
        y_cols (list): A list of column names from `df` that should
                                   be used as y variables (future values).
        lookforward_timedelta (pd.Timedelta): The duration for the look-forward window.
                                           e.g., pd.Timedelta(hours=1) for 1 hour lookforward.
        
    Returns:
        tuple:
            - output_array (np.ndarray): Shape (num_samples, lookforward_timestep, num_target_cols).
                                        Contains lagged features for each sample.
            - datetime (pd.DatetimeIndex): The datetime index corresponding to the
                                           start of each valid sample (after NaN removal).
    """

    output_array = []
    for i in range(lookforward_timestep):
        shifted_col_numpy = df[target_col].shift(-i).to_numpy()
        output_array.append(shifted_col_numpy)
    output_array = np.array(output_array).transpose(1, 0)              # input_array has a shape (num_sample, lookforward_timestep, num_target_col)

    datetime = df.index

    return output_array, datetime

def scale_transform_single_var(input_array):
    """
    Scales the input NumPy array using StandardScaler.
    It can handle both 2D (samples, features) and 3D (samples, timesteps, features) arrays.
    For 3D arrays, it scales each feature independently across all samples and timesteps.

    Args:
        input_array (np.ndarray): The NumPy array to be scaled.
                                  Can be 2D (num_samples, num_features)
                                  or 3D (num_samples, lookback_timestep, num_features).

    Returns:
        tuple:
            - scaled_array (np.ndarray): The scaled version of the input array.
            - scalers (list of StandardScaler or StandardScaler): A list of StandardScaler
              objects (one for each feature in 3D, or a single scaler in 2D)
              used for the transformation. These are needed for inverse transformation.
        None: If the input_array has a dimension other than 2 or 3.
    """
    scalers = []
    if input_array.ndim == 2:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(input_array)
        scalers.append(scaler)

    elif input_array.ndim == 3:
        scalers = []
        scaled_array = np.zeros(input_array.shape)
        for i in range(input_array.shape[-1]):
            scaler = StandardScaler()
            scaled_array[:, :, i] = scaler.fit_transform(input_array[:, :, i])
            scalers.append(scaler)

    else: 
        return None
    
    return scaled_array, scalers

def inverse_scale_transform_single_var(input_scaled_array, scalers):
    """
    Performs the inverse scaling transformation on a scaled NumPy array
    using the provided StandardScaler objects.
    It can handle both 2D and 3D scaled arrays.

    Args:
        input_scaled_array (np.ndarray): The scaled NumPy array to be inverse transformed.
        scalers (list of StandardScaler or StandardScaler): The scaler(s) used for the
                                                          original transformation.
                                                          If 2D, can be a single scaler or a list containing one.
                                                          If 3D, must be a list of scalers (one per feature).

    Returns:
        np.ndarray: The original (unscaled) NumPy array.
    """
    if input_scaled_array.ndim == 2:
        if isinstance(scalers, list): scaler = scalers[0]
        else: scaler = scalers
        original_array = scaler.inverse_transform(input_scaled_array)

    elif input_scaled_array.ndim == 3:
        original_array = np.zeros(input_scaled_array.shape)
        for i in range(input_scaled_array.shape[-1]):
            original_array[:, :, i] = scalers[i].inverse_transform(input_scaled_array[:, :, i])
    
    else: return None

    return original_array

def split_time_series_multi_feature(x_lag, x_exo, y, datetime, train_ratio, val_ratio):
    if x_lag.shape[0] != x_exo.shape[0]: return None
    total_samples = x_lag.shape[0]

    train_samples_count = int(total_samples * train_ratio)
    val_samples_count = int(total_samples * val_ratio)
    test_samples_count = total_samples - train_samples_count - val_samples_count

    # Chronological split
    x_lag_train = x_lag[:train_samples_count]
    x_exo_train = x_exo[:train_samples_count]
    y_train = y[:train_samples_count]

    x_lag_val = x_lag[train_samples_count : train_samples_count + val_samples_count]
    x_exo_val = x_exo[train_samples_count : train_samples_count + val_samples_count]
    y_val = y[train_samples_count : train_samples_count + val_samples_count]

    x_lag_test = x_lag[train_samples_count + val_samples_count :]
    x_exo_test = x_exo[train_samples_count + val_samples_count :]
    y_test = y[train_samples_count + val_samples_count :]

    # return datetime
    datetime_train = datetime[:train_samples_count]
    datetime_val = datetime[train_samples_count : train_samples_count + val_samples_count]
    datetime_test = datetime[train_samples_count + val_samples_count :]

    print(f"\n--- Data Split Summary ---")
    print(f"Total samples:      {total_samples}")
    print(f"Train samples:      {x_lag_train.shape[0]}   (X_lag shape: {x_lag_train.shape}, X_exo shape: {x_exo_train.shape}, Y shape: {y_train.shape})")
    print(f"Validation samples: {x_lag_val.shape[0]}     (X_lag shape: {x_lag_val.shape}, X_exo shape: {x_exo_val.shape}, Y shape: {y_val.shape})")
    print(f"Test samples:       {x_lag_test.shape[0]}    (X_lag shape: {x_lag_test.shape}, X_exo shape: {x_exo_test.shape}, Y shape: {y_test.shape})")
    print(f"--------------------------")

    return (x_lag_train, x_exo_train, y_train), (x_lag_val, x_exo_val, y_val), (x_lag_test, x_exo_test, y_test), (datetime_train, datetime_val, datetime_test)

# # --------------------------------------- Import Models --------------------------------------- #

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # d_model must be even
        d_model_even = 2 * ((d_model + 1) // 2)

        pe = torch.zeros(max_len, d_model_even)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe[:, :d_model]
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        
        x = x + self.pe[:, :x.size(1), :]  # Ensure correct slicing

        return self.dropout(x)          # avoid overfitting with dropout regularization
    
class TransformerEncoderModel(nn.Module):
    """
    A Transformer Encoder model for sequence-to-one regression tasks.
    """
    def __init__(self,
                 input_size: int = 1,
                 output_size: int = 1,
                 seq_len: int = 200,
                 d_model: int = 16,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropput = dropout

        # --- Embedding and Positional Encoding ---
        # Linear layer to embed the input dimension into the model's dimension
        self.input_embedding  = nn.Linear(input_size, d_model)
        # Adds positional information to the embedded sequence
        self.position_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len)

        # --- Transformer Encoder ---
        # Define a single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Crucial for (batch, seq_len, feature) shape
        )
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Regression Head ---
        # A series of linear layers to map the transformer's output to the final prediction
        self.regression_head = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size),
        )

    # Model Forward Pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [batch_size, seq_len, input_size]
        """
        # 1. Embed the input and add positional encoding
        x = self.input_embedding(x)
        x = self.position_encoder(x)

        # 2. Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # 3. Flatten the output for the regression head
        x = x.reshape(-1, self.seq_len * self.d_model)

        # 4. Pass through the regression head
        x = self.regression_head(x)

        return x   
    
# --- Decoder-Only Transformer Model ---
class TransformerDecoderModel(nn.Module):
    """
    A Decoder-Only Transformer model for sequence-to-one regression tasks.
    """
    def __init__(self,
                 input_size: int = 1,
                 output_size: int = 1,
                 seq_len: int = 200,
                 d_model: int = 16,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 64,
                 dropout: int = 0.1,
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropput = dropout

        # --- Embedding and Positional Encoding ---
        # Linear layer to embed the input dimension into the model's dimension
        self.input_embedding  = nn.Linear(input_size, d_model)
        # Adds positional information to the embedded sequence
        self.position_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len)

        # --- Transformer Decoder ---
        # Define a single decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Crucial for (batch, seq_len, feature) shape
        )
        # Stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- Regression Head ---
        # A series of linear layers to map the transformer's output to the final prediction
        self.regression_head = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size)
        )

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        """
        Generates a square causal mask for the sequence.
        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        This prevents the model from attending to future tokens.
        """
        return nn.Transformer.generate_square_subsequent_mask(sz)

    # Model Forward Pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [batch_size, seq_len, input_size]
        """
        # 1. Generate the causal mask and move it to the correct device
        # The mask ensures that a position can only attend to itself and previous positions.
        causal_mask = self._generate_causal_mask(self.seq_len).to(x.device)

        # 2. Embed the input and add positional encoding
        x = self.input_embedding(x)
        x = self.position_encoder(x)

        # 3. Pass through the transformer decoder
        # For a decoder-only architecture, the target (tgt) and memory are the same.
        # We apply the causal mask to the target sequence attention.
        x = self.transformer_decoder(tgt=x, memory=x, tgt_mask=causal_mask)

        # 4. Flatten the output for the regression head
        x = x.reshape(-1, self.seq_len * self.d_model)

        # 5. Pass through the regression head
        x = self.regression_head(x)

        return x
    
# --- Encoder-Decoder Transformer Model ---
class TransformerEncoderDecoderModel(nn.Module):
    """
    A standard Encoder-Decoder Transformer model for sequence-to-sequence tasks.
    """
    def __init__(self,
                 input_size: int = 1,
                 output_size: int = 1,
                 seq_len: int = 200,
                 d_model: int = 16,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropput = dropout

        self.output_size = output_size

        # --- Core Transformer ---
        # Using PyTorch's built-in nn.Transformer module which contains both encoder and decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Ensures input/output shapes are (batch, seq, feature)
        )

        # --- Embedding and Positional Encoding ---
        # Linear layers to embed the source and target sequences into the model's dimension
        self.source_embedding = nn.Linear(input_size, d_model)
        self.target_embedding = nn.Linear(output_size, d_model) # Target input embedding

        # Adds positional information to the embedded sequences
        self.src_position_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len)
        self.tgt_position_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=1)

         # --- Regression Head ---
        # A series of linear layers to map the transformer's output to the final prediction
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size)
        )

        # # --- Final Output Layer ---
        # # Maps the decoder's output from d_model to the desired output dimension
        # self.output_layer = nn.Linear(d_model, output_size)

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        """
        Generates a square causal mask for the sequence.
        This prevents the decoder from attending to future tokens during training.
        """
        return nn.Transformer.generate_square_subsequent_mask(sz)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Source sequence, shape [batch_size, src_seq_len, input_size]
            tgt: Target sequence, shape [batch_size, tgt_seq_ln, output_size]
        """
        
        # --- FIXED LOGIC: Use a single variable for the decoder input ---
        if tgt is None:
            # Inference mode: use a placeholder as the decoder input
            batch_size = src.size(0)
            tgt_seq_input = torch.zeros(batch_size, 1, self.output_size, device=src.device)
        else:
            # Training mode: use the teacher-forced target as the decoder input
            if tgt.dim() == 2:
                # Reshape target from [batch, output_size] to [batch, 1, output_size]
                tgt_seq_input = tgt.unsqueeze(1)
            else:
                # If it already has the sequence dimension, use it as is
                tgt_seq_input = tgt

        # 1. Generate masks
        # Use the sequence length of the *reshaped* decoder input
        tgt_seq_len = tgt_seq_input.size(1)
        tgt_mask = self._generate_causal_mask(tgt_seq_len).to(src.device)

        # 2. Embed source and target sequences and add positional encoding
        # Use the `tgt_seq_input` variable consistently
        src_embedded = self.src_position_encoder(self.source_embedding(src))
        tgt_embedded = self.tgt_position_encoder(self.target_embedding(tgt_seq_input))

        # print(f"src_embedded shape: {src_embedded.shape}, tgt_embedded shape: {tgt_embedded.shape}")
        
        # 3. Pass through the transformer
        x = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            tgt_mask=tgt_mask
        )

        # 4. Pass through the regression head
        x = self.regression_head(x[:, -1, :]) # Use only the last token's output for regression

        return x


# Hybrid model
class HybridNetModel(nn.Module):
    def __init__(self, 
                 x_lag_input_size: int = 1,
                 x_exo_input_size: int = 1,
                 output_size: int = 1,
                 seq_len: int = 200,
                 d_model: int = 16,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 ):
        super().__init__()  

        self.seq_len = seq_len
        self.d_model = d_model
        self.output_size = output_size

        # Embedding input lag regressors
        self.input_embedding = nn.Linear(x_lag_input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use (batch, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(seq_len * d_model + x_exo_input_size, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size)
        )

    def forward(self, x_lag, x_exo):
        """
        Args:
            lag_regressors: Tensor [B, T, F] = [batch, seq_len, lag_feature_dim]
            exogenous_variables: Tensor [B, E] = [batch, exo_feature_dim]
        Returns:
            Tensor [B, output_size]
        """
        # Embed + position encode
        x = self.input_embedding(x_lag)               # [B, T, d_model]
        x = self.positional_encoding(x)                        # [B, T, d_model]
        x = self.transformer_encoder(x)                        # [B, T, d_model]

        # Flatten transformer output
        x = x.reshape(x.size(0), -1)                           # [B, T*d_model]

        # Concatenate with exogenous variables
        x = torch.cat([x, x_exo], dim=1)        # [B, T*d_model + E]

        # Predict output
        out = self.regression_head(x)                         # [B, output_size]
        return out

# --------------------------------------- Train models --------------------------------------- #

def plot_losses(loss_history, val_loss_history=None, gradient_plot=True):
    epochs = np.arange(len(loss_history))
    fig, ax = plt.subplots(2 if gradient_plot else 1, 1, figsize=(10, 8 if gradient_plot else 4))

    if not isinstance(ax, np.ndarray):
        ax = [ax]

    ax[0].plot(loss_history, label='Training Loss', color='dodgerblue')
    if val_loss_history:
        ax[0].plot(val_loss_history, label='Validation Loss', linestyle='--', color='orangered')
    ax[0].set_title("Training and Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[0].legend()

    if gradient_plot:
        loss_grad = np.gradient(loss_history)
        val_grad = np.gradient(val_loss_history) if val_loss_history else []

        ax[1].plot(epochs, loss_grad, label='Training Loss Gradient', color='dodgerblue')
        if val_loss_history:
            ax[1].plot(epochs, val_grad, label='Validation Loss Gradient', linestyle='--', color='orangered')
        ax[1].set_title("Gradient of Loss Curve")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Gradient")
        ax[1].grid(True)
        ax[1].legend()

    plt.tight_layout()
    plt.show()
    # return fig

def trainer_multi_feature(model, train_dataloader, val_dataloader, optimizer=Adam, criterion=nn.MSELoss(), scheduler=None, max_epochs=200, early_stopping_patience=-1, display_result=True, device=None):
    if device == None: device = cuda_check()
    if isinstance(optimizer, type): optimizer = optimizer(model.parameters(), lr=0.001)

    loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch = -1

    early_stop_count = 0

    if display_result:
        print("\n--- Training ---")
        print("Training started...")

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for x_lag_batch, x_exo_batch, y_batch in train_dataloader:
            x_lag_batch, x_exo_batch, y_batch = x_lag_batch.to(device), x_exo_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_lag_batch, x_exo_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        loss_history.append(avg_train_loss)

        # ----- Validation -----
        avg_val_loss = np.nan
        additional_str = ''

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_lag_batch, x_exo_batch, y_batch in val_dataloader:
                x_lag_batch, x_exo_batch, y_batch = x_lag_batch.to(device), x_exo_batch.to(device), y_batch.to(device)
                y_pred = model(x_lag_batch, x_exo_batch)
                loss = criterion(y_pred, y_batch) 
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_loss_history.append(avg_val_loss)

        # Scheduler update
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, ExponentialLR):
            scheduler.step()

        # Early stopping & best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())      # Copy the best model
            early_stop_count = 0
            additional_str = "Best model updated"
        else:
            early_stop_count += 1
            if early_stopping_patience != -1 and early_stop_count >= early_stopping_patience:
                if display_result:
                    print("Early stopping triggered!")
                break

        if display_result:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | {additional_str}")

    if display_result:
        print("\nTraining completed.")
        if best_epoch >= 0:
            print(f"Best model at epoch {best_epoch + 1} with Val Loss: {best_val_loss:.4f}")
        else:
            print("No validation improvement occurred.")

        plot_losses(loss_history, val_loss_history, gradient_plot=True)

    # Load the best model before returning
    model.load_state_dict(best_model_weights)

    return model, loss_history, val_loss_history


def predictor_multi_feature(model, test_dataloader, y_scaler, criterion=nn.MSELoss(), device=cuda_check()):
    model.eval()
    y_pred_scaled = []
    y_test_scaled = []
    test_epoch_loss = 0

    with torch.no_grad():
        for x_lag_batch, x_exo_batch, y_batch in test_dataloader:
            x_lag_batch, x_exo_batch, y_batch = x_lag_batch.to(device), x_exo_batch.to(device), y_batch.to(device)
            y_pred = model(x_lag_batch, x_exo_batch)
            loss = criterion(y_pred, y_batch) 
            test_epoch_loss += loss.item()
            y_pred_scaled.append(y_pred.cpu())
            y_test_scaled.append(y_batch.cpu())

    y_pred_scaled = torch.cat(y_pred_scaled, dim=0)
    y_test_scaled = torch.cat(y_test_scaled, dim=0)
    eval_loss = test_epoch_loss / len(test_dataloader)

    y_test = y_scaler.inverse_transform(y_test_scaled.cpu().numpy())
    y_pred = y_scaler.inverse_transform(y_pred_scaled.cpu().numpy())

    return y_pred, y_test, eval_loss


# --------------------- Optuna Hyperparameter Optimization ------------------------------ #
# Copy these objective functions to the notebook to specify search spaces

def objective_bayesian(trial, model_parameters, train_dataloader, val_dataloader, device=cuda_check()):
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    try:
        d_model_per_head = trial.suggest_categorical('d_model_per_head', [4, 8, 16, 32, 64])
        nhead = trial.suggest_categorical('nhead', [1, 2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512, 1024])
            
        model = TransformerEncoderModel(
            input_size=model_parameters['num_feature'],
            output_size=model_parameters['lookforward_timestep'],
            seq_len=model_parameters['lookback_timestep'],
            d_model=d_model_per_head * nhead,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1
        ).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()
        # scheduler = ExponentialLR(optimizer, gamma=0.99)

        _, _, val_loss = trainer_multi_feature(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            scheduler=None,
            max_epochs=50,
            early_stopping_patience=20,
            display_result=False
        )

        return val_loss[-1] if val_loss else float('inf')

    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')
    

def objective_grid(trial, model_parameters, train_dataloader, val_dataloader, device=cuda_check()):
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    try:
        d_model_per_head = trial.suggest_categorical('d_model_per_head', [4, 8, 16, 32, 64])
        nhead = trial.suggest_categorical('nhead', [1, 2, 4, 8])
        num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4])
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512, 1024])

        model = TransformerEncoderModel(
            input_size=model_parameters['num_feature'],
            output_size=model_parameters['lookforward_timestep'],
            seq_len=model_parameters['lookback_timestep'],
            d_model=d_model_per_head * nhead,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1
        ).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()
        # scheduler = ExponentialLR(optimizer, gamma=0.99)

        _, _, val_loss = trainer_multi_feature(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            scheduler=None,
            max_epochs=200,
            early_stopping_patience=20,
            display_result=False
        )

        return val_loss[-1] if val_loss else float('inf')

    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')
    
# -------------------------------------------------------------------------------------------- #

 # Calculate total number of trials (all combinations) for Grid Search
def num_combination_grid(search_space):
    n_trials = reduce(mul, [len(v) for v in search_space.values()])  
    return n_trials 


def run_optimization(objective, n_trials, sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3), timeout=None, n_jobs=5):
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler, 
        pruner=pruner,
    )

    # if timeout == None: timeout = 180 * n_trials

    study.optimize(objective, n_trials=n_trials, n_jobs = n_jobs, timeout=timeout, show_progress_bar=True)

    print('\n--- Optimization Complete ---')
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Validation Loss: {study.best_trial.value:.6f}")
    print("Best Parameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")

    return study


def evaluate_model(model, test_dataloader, device=cuda_check()):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in test_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return r2_score(all_targets, all_preds), mean_squared_error(all_targets, all_preds), mean_absolute_error(all_targets, all_preds)


def train_final_model(study, model_parameters, train_dataloader, val_dataloader, test_dataloader, optimizer=Adam, criterion=nn.MSELoss(), scheduler=None, max_epochs=200, early_stopping_patience=5, seed=42, device=cuda_check()):
    best_params = study.best_params

    torch.manual_seed(seed)
    final_model = TransformerEncoderModel(
        input_size=model_parameters['num_feature'],
        output_size=model_parameters['lookforward_timestep'],
        seq_len=model_parameters['lookback_timestep'],
        d_model=best_params['d_model_per_head'] * best_params['nhead'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers'],
        dim_feedforward=best_params['dim_feedforward'],
        dropout=0.1
    ).to(device)

    final_model, train_loss, val_loss = trainer_multi_feature(
        model=final_model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scheduler=scheduler,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience
    )

    r2, mse, mae = evaluate_model(final_model, test_dataloader)

    print("\n--- Final Model Evaluation ---")
    print(f"Test R² Score: {r2:.4f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")

    return final_model, r2, mse, mae

def show_n_best_models(study, n, verbose = 1):
    # After study.optimize() completes
    all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Sort trials by their validation loss (ascending)
    sorted_trials = sorted(all_trials, key=lambda t: t.value)   
    model_parameters = []

    for i in range(n):
        trial = sorted_trials[i]
        model_parameters.append(trial)
        if verbose==1:
            print(f"Rank {i+1} Trial:")
            print(f"  Number: {trial.number}")
            print(f"  Value (Loss): {trial.value}")
            print(f"  Params: {trial.params}")
    return model_parameters

# --------------------- Save Model --------------------------------- #

def save_HybridNetModel(model, path):
    model_params = {
    'input_size': model.input_size,
    'output_size': model.output_size,
    'seq_len': model.seq_len,
    'd_model':model.d_model, 
    'nhead': model.nhead, 
    'num_layers': model.num_layers, 
    'dim_feedforward': model.dim_feedforward, 
    'dropout': model.dropput, 
    }
    torch.save({
        'model_state': model.state_dict(), 
        'model_params': model_params, 
        }, path)
    
    print(f"Model state dictionary saved to: {path}")

# --------------------- Load Model --------------------------------- #

def load_HybridNetModel(model, path):
    # model must be MODEL not MODEL(...)
    # To prevent an error from typing MODEL() 
    if isinstance(model, type): model = model
    else: model =  model.__class__
    
    load = torch.load(path, map_location='cpu')
    params = load['model_params']
    model_state = load['model_state']

    model = model(**params)

    if os.path.exists(path):
        model.load_state_dict(model_state)
        print('Loading complete')
        return model
    
    else:
        print(f"Error: Model state dictionary file not found at {path}")
        return None

# --------------------- Visualization ------------------------------ #

def plotly_visualization_line(x, y, name=[]):
    if len(y) != len(name): return 'len(y) != len(name)'
    if len([x]) == 1: 
        x_extended = []
        for i in range(len(y)): x_extended.append(x)
        x = x_extended

    # Visualize the result
    fig = go.Figure()

    # Add the traces
    for i in range(len(y)):
        fig.add_trace(go.Scatter(x=x[i], y=y[i],
                         mode='lines',
                         name=name[i],
                         line=dict(width=1))
                         )
        
    # Update layout for title, axis labels, and grid
    fig.update_layout(
        title='Time Series Plot',
        xaxis_title='DateTime',
        yaxis_title='Irradiance',
        hovermode='x unified', # This is useful for time series to see values across traces at a given x
        template='plotly_white' # A clean white background template
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    fig.show()