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

def df_transform(df: pd.DataFrame, lag_regressor_cols: list, target_cols: list, lookback_timedelta: pd.Timedelta = pd.Timedelta(minutes=0), lookforward_timedelta: pd.Timedelta = pd.Timedelta(minutes=0), return_df = False):
    
    if len(df.index) < 2:
        raise ValueError("DataFrame must have at least two entries to determine frequency.")
    time_interval = df.index[1] - df.index[0]
    if time_interval == pd.Timedelta(minutes=0):
        raise ValueError("Time interval between consecutive DataFrame indices is zero, cannot calculate timesteps.")

    lookback_timestep = lookback_timedelta // time_interval
    lookforward_timestep = lookforward_timedelta // time_interval
    num_feature = df.shape[1]
    num_target = len(target_cols)

    # Preparing y(t), y(t-1), ..., y(t-L), L = lookback_timestep
    input_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        temp_df = pd.DataFrame(index=df.index)
        for i in range(lookback_timestep):
            shift = i + 1 if col in lag_regressor_cols else i
            shifted_col = df[col].shift(shift)
            temp_df[i] = shifted_col
        input_df = pd.concat([input_df, temp_df], axis=1)
    
    # Preparing y(t+1), ..., y(t+M), M = lookforward_step
    output_df = pd.DataFrame(index=df.index)
    for col in target_cols:
        temp_df = pd.DataFrame(index=df.index)
        for i in range(1, lookforward_timestep + 1):
            shifted_col = df[col].shift(-i)
            temp_df[i] = shifted_col.values
        output_df = pd.concat([output_df, temp_df], axis=1)
    
    all_df = pd.concat([input_df, output_df], axis=1).dropna()
    datetime = all_df.index
    input_df = input_df.loc[all_df.index]
    output_df = output_df.loc[all_df.index]

    input_arrays = input_df.to_numpy().reshape(-1, lookback_timestep, num_feature)
    output_arrays = output_df.to_numpy().reshape(-1, num_target * lookforward_timestep)

    input_labels = list(df.columns)
    output_labels = target_cols

    return input_arrays, output_arrays, input_labels, output_labels, datetime

def df_transform_numpy(df: pd.DataFrame, lag_regressor_cols: list, target_cols: list, lookback_timedelta: pd.Timedelta = pd.Timedelta(minutes=0), lookforward_timedelta: pd.Timedelta = pd.Timedelta(minutes=0), return_df = False):
    
    if len(df.index) < 2:
        raise ValueError("DataFrame must have at least two entries to determine frequency.")
    time_interval = df.index[1] - df.index[0]
    if time_interval == pd.Timedelta(minutes=0):
        raise ValueError("Time interval between consecutive DataFrame indices is zero, cannot calculate timesteps.")

    lookback_timestep = lookback_timedelta // time_interval
    lookforward_timestep = lookforward_timedelta // time_interval
    num_feature = df.shape[1]
    num_target = len(target_cols)

    # Preparing y(t), y(t-1), ..., y(t-L), L = lookback_timestep
    input_arrays = []
    for col in df.columns:
        lagged = []
        for i in range(lookback_timestep):
            shift = i + 1 if col in lag_regressor_cols else i
            shifted_col = df[col].shift(shift)
            lagged.append(shifted_col.values)
        input_arrays.append(lagged)
    input_arrays = np.transpose(np.array(input_arrays), (2, 1, 0))

    # Preparing y(t+1), ..., y(t+M), M = lookforward_step
    output_arrays = []
    for col in target_cols:
        future = []
        for i in range(1, lookforward_timestep + 1):
            shifted_col = df[col].shift(-i)
            future.append(shifted_col.values)
        output_arrays.append(future)
    output_arrays = np.transpose(np.array(output_arrays), (2, 1, 0)).reshape(-1, num_target * lookforward_timestep)

    input_mask = ~np.isnan(input_arrays).any(axis=(1, 2))
    output_mask = ~np.isnan(output_arrays).any(axis=1)

    all_mask = input_mask & output_mask

    input_arrays = input_arrays[all_mask, :, :]
    output_arrays = output_arrays[all_mask, :]

    datetime = df[all_mask].index
    input_labels = list(df.columns)
    output_labels = target_cols

    return input_arrays, output_arrays, input_labels, output_labels, datetime


def split_time_series_data(x: torch.Tensor, y: torch.Tensor, train_ratio: float, val_ratio: float, datetime: list):
    """
    Splits time series data (X and Y PyTorch tensors) into training,
    validation, and test sets chronologically based on specified ratios.

    Args:
        x (torch.Tensor): Input features tensor.
        y (torch.Tensor): Target values tensor.
        train_ratio (float): Proportion of data for the training set (e.g., 0.6).
        val_ratio (float): Proportion of data for the validation set (e.g., 0.2).

    Returns:
        tuple: A tuple containing six PyTorch tensors:
               (x_train, y_train, x_val, y_val, x_test, y_test)

    Raises:
        ValueError: If ratios do not sum up to 1 or if data is too short for splits.
    """
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and (train_ratio + val_ratio) <= 1):
        raise ValueError("Train and validation ratios must be between 0 and 1, and their sum must be <= 1.")

    total_samples = x.shape[0]
    if total_samples == 0:
        raise ValueError("No samples to split. Ensure your to_sequence function produced data.")

    train_samples_count = int(total_samples * train_ratio)
    val_samples_count = int(total_samples * val_ratio)
    test_samples_count = total_samples - train_samples_count - val_samples_count

    # Chronological split
    x_train = x[:train_samples_count]
    y_train = y[:train_samples_count]

    x_val = x[train_samples_count : train_samples_count + val_samples_count]
    y_val = y[train_samples_count : train_samples_count + val_samples_count]

    x_test = x[train_samples_count + val_samples_count :]
    y_test = y[train_samples_count + val_samples_count :]

    # return datetime
    datetime_train = datetime[:train_samples_count]
    datetime_val = datetime[train_samples_count : train_samples_count + val_samples_count]
    datetime_test = datetime[train_samples_count + val_samples_count :]

    print(f"\n--- Data Split Summary ---")
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {x_train.shape[0]} (X shape: {x_train.shape}, Y shape: {y_train.shape})")
    print(f"Validation samples: {x_val.shape[0]} (X shape: {x_val.shape}, Y shape: {y_val.shape})")
    print(f"Test samples: {x_test.shape[0]} (X shape: {x_test.shape}, Y shape: {y_test.shape})")
    print(f"--------------------------")

    return x_train, y_train, x_val, y_val, x_test, y_test, datetime_train, datetime_val, datetime_test


def scale_transform(x_array: np.ndarray, y_array: np.ndarray):
    """
    Scale 3D input/output arrays using StandardScaler per feature/target.
    Assumes x_numpy: [N, Cx, Tx], y_numpy: [N, Cy, Ty]

    Returns:
        x_scaled (torch.Tensor): Scaled input tensor [N, Cx, Tx]
        y_scaled (torch.Tensor): Scaled target tensor [N, Cy, Ty]
        x_scalers (dict): Per-feature scalers for inverse_transform
        y_scalers (dict): Per-target scalers
    """

    num_sample, num_feature, lookback_timestep = x_array.shape
    x_scaled_array = np.zeros_like(x_array)
    x_scalers = []

    for i in range(num_feature):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(x_array[:, i, :])
        x_scaled_array[:, i, :] = scaled
        x_scalers.append(scaler)

    y_scaler = StandardScaler()
    y_scaled_array = y_scaler.fit_transform(y_array)
    
    return x_scaled_array, y_scaled_array, x_scalers, y_scaler

def inverse_scale_transform(scaled_numpy: np.ndarray, scalers: list) -> np.ndarray:
    """
    Inverse transform a scaled 3D tensor using a list of fitted scalers.
    Assumes input shape is [N, C, T].

    Args:
        scaled_tensor (torch.Tensor): Scaled tensor of shape [N, C, T] (on any device).
        scalers (list): List of fitted StandardScaler objects (length = C).

    Returns:
        np.ndarray: Inverse-transformed array of shape [N, C, T].
    """
    num_sample, num_feature, lookback_timestep = scaled_numpy.shape
    output_array = np.zeros_like(scaled_numpy)

    for i in range(num_feature):
        inversed = scalers[i].inverse_transform(scaled_numpy[:, i, :])
        output_array[:, i, :] = inversed

    return output_array

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
            # nn.ReLU(),
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

def trainer(model, optimizer, train_dataloader, val_dataloader, criterion=nn.MSELoss(), 
            scheduler=None, max_epochs=200, early_stopping_patience=-1, display_result=True, device=None,
):
    """
    Trains a PyTorch model with optional validation, early stopping, and scheduler support.

    Args:
        model (nn.Module): Model to train.
        optimizer (torch.optim.Optimizer or type): Optimizer instance or class.
        train_dataloader (DataLoader): Training data.
        criterion (loss function): Loss function.
        val_dataloader (DataLoader, optional): Validation data.
        scheduler (lr scheduler, optional): Learning rate scheduler.
        max_epochs (int): Number of training epochs.
        early_stopping_patience (int): Stop if no improvement after this many epochs. Set -1 to disable.
        display_result (bool): Whether to print logs and plot losses.
        device (torch.device, optional): Device to use. Auto-detects if None.

    Returns:
        model (nn.Module): Best model based on validation loss.
        loss_history (list): Training loss history.
        val_loss_history (list): Validation loss history.
    """
    if device is None:
        device = cuda_check()

    if isinstance(optimizer, type):
        optimizer = optimizer(model.parameters(), lr=0.001)

    model.to(device)
    loss_history, val_loss_history = [], []

    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch = -1
    early_stop_counter = 0

    if display_result:
        print("\n--- Training Started ---")

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_dataloader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        loss_history.append(avg_train_loss)

        # --- Validation Phase ---
        avg_val_loss = float('nan')
        additional_note = ''

        
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_dataloader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                val_loss += criterion(y_pred, yb).item()

        avg_val_loss = val_loss / len(val_dataloader)
        val_loss_history.append(avg_val_loss)

        # Scheduler step
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, ExponentialLR):
            scheduler.step()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            early_stop_counter = 0
            additional_note = "Best model updated"
        else:
            early_stop_counter += 1
            if early_stopping_patience != -1 and early_stop_counter >= early_stopping_patience:
                if display_result:
                    print("Early stopping triggered!")
                break

        if display_result:
            print(
                f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} {additional_note}"
            )

    # Restore best model
    model.load_state_dict(best_model_weights)

    if display_result:
        print("\n--- Training Complete ---")
        if best_epoch >= 0:
            print(f"Best Epoch: {best_epoch + 1} | Best Val Loss: {best_val_loss:.4f}")
        else:
            print("Validation not used or no improvement.")
        plot_losses(loss_history, val_loss_history)

    return model, loss_history, val_loss_history

def predictor(model, test_dataloader, y_scaler, criterion=nn.MSELoss(), device=None):
    """
    Evaluate the model on the test set and return predictions and evaluation loss.

    Args:
        model (nn.Module): Trained model.
        test_dataloader (DataLoader): DataLoader for test data.
        y_scaler (sklearn scaler): Scaler used for target inverse transform.
        criterion (loss function): Loss function.
        device (torch.device, optional): Device for evaluation.

    Returns:
        y_pred (ndarray): Denormalized predicted values.
        y_true (ndarray): Denormalized ground-truth values.
        eval_loss (float): Average loss on test set.
    """
    if device is None:
        device = cuda_check()

    model.eval()
    y_preds, y_trues = [], []
    test_loss = 0.0

    with torch.no_grad():
        for xb, yb in test_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            test_loss += criterion(y_pred, yb).item()
            y_preds.append(y_pred.cpu())
            y_trues.append(yb.cpu())

    y_preds = torch.cat(y_preds, dim=0).numpy()
    y_trues = torch.cat(y_trues, dim=0).numpy()

    y_pred = y_scaler.inverse_transform(y_preds)
    y_true = y_scaler.inverse_transform(y_trues)
    eval_loss = test_loss / len(test_dataloader)

    return y_pred, y_true, eval_loss


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

        _, _, val_loss = trainer(
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

        _, _, val_loss = trainer(
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

    final_model, train_loss, val_loss = trainer(
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

def save_model(model, path):
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

def load_model(model, path):
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