import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime
import math
import re
from scipy.stats import linregress
import json
import sys
# from torcheval.metrics import R2Score
# from torchmetrics.functional import r2_score


class SNN_GPP_Tram(nn.Module):
    def __init__(self, input_dim, hidden_layer_size):
        super(SNN_GPP_Tram, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, 1),
            nn.Sigmoid(),
            # TODO: NEED TO LATER ON MULTIPLY THE OUTPUT OF THIS WITH SW_IN, THEN PUSH IT THROUGH POSLIN.
        )

    def forward(self, x):
        return self.net(x)

class SNN_RECO_Tram(nn.Module):
    def __init__(self, input_dim, hidden_layer_size):
        super(SNN_RECO_Tram, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)



class SNN_GPP(nn.Module):
    def __init__(self, input_dim, hidden_layer_size):
        super(SNN_GPP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
            nn.ReLU() # WORKS!
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

class SNN_RECO(nn.Module):
    def __init__(self, input_dim, hidden_layer_size):
        super(SNN_RECO, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
            nn.ReLU() # WORKS!
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    # Ensure both tensors are on the same device
    y_true = y_true.to(y_pred.device)

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    # Ensure both tensors are on the same device
    y_true = y_true.to(y_pred.device)

    mse = torch.mean((y_true - y_pred) ** 2)
    return torch.sqrt(mse).item()

def better_fit_gpu(X_gpp_train, X_reco_train, y_train, 
        X_gpp_val, X_reco_val, y_val,
        SW_IN_RAW_train, SW_IN_RAW_val,
        tram = False,
        epochs=100000, lr=1e-3,
        hidden_layer_size = 12
        ):

    run_info = "Tramontana" if tram else "Custom"

    # Early stopping conditions
    patience = 500
    min_delta = 1e-2

    # __device = torch.device("cuda" if torch.cuda.is_available() else "cpu")__
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(run_info, "Running on GPU", "With hidden layer size of", hidden_layer_size, "Total epochs:", epochs, "lr:", lr, "patience:", patience, "min_delta", min_delta)
    else:
        print(run_info, "Running on CPU", "With hidden layer size of", hidden_layer_size, "Total epochs:", epochs, "lr:", lr, "patience:", patience, "min_delta", min_delta)

    # __Move data to device__
    X_gpp_train = X_gpp_train.to(device)
    X_reco_train = X_reco_train.to(device)
    y_train = y_train.to(device)
    X_gpp_val = X_gpp_val.to(device)
    X_reco_val = X_reco_val.to(device)
    y_val = y_val.to(device)
    if tram:
        SW_IN_RAW_train = SW_IN_RAW_train.to(device)
        SW_IN_RAW_val = SW_IN_RAW_val.to(device)
    # Else these variables do not need to be moved to the GPU.

    # Instantiate models
    if tram:
        gpp_model = SNN_GPP_Tram(X_gpp_train.shape[1], hidden_layer_size)
        reco_model = SNN_RECO_Tram(X_reco_train.shape[1], hidden_layer_size)
    else:
        gpp_model = SNN_GPP(X_gpp_train.shape[1], hidden_layer_size)
        reco_model = SNN_RECO(X_reco_train.shape[1], hidden_layer_size)

    # Show what the model looks like for output tracking purposes.
    print(gpp_model)
    print(reco_model)

    # Move models to device
    gpp_model = gpp_model.to(device)
    reco_model = reco_model.to(device)

    # Optimizer
    optimizer = optim.Adam(list(gpp_model.parameters()) + list(reco_model.parameters()), lr=lr)

    # Loss function
    criterion = nn.MSELoss()

    # Early stopping variables:
    best_val_r2 = -float('inf')
    epochs_since_improvement = 0

    # Early stopping variables:
    # min_slope is the minimum acceptable upward slope in validation
    #   and it measured every {trend_window} many epochs
    # --- Parameters ---
    if not tram: # ie custom run. We know the following numbers work really well with full feature set. AND also reduced sets too!
        # min_slope = 0.001
        # min_slope = 0.00001
        # min_slope = 1e-6 #  # Minimum acceptable upward slope in validation
        trend_window = 10  # Number of past epochs to use for trend analysis
        min_slope = 5e-6 # best 95 93 79 85
        trend_window = 100
        min_slope = 2e-6 # #95 93 74 76 Tram - changing this to higher number just leads to overfitting.
        print("Custom will use the Tram vars")
    else: # the Tramontana model takes much longer to train.
        # trend_window = 500 #95 94 71 73 pretty close
        # trend_window = 50 # 95 93 71 70
        # min_slope = 5e-7 # r2 .9306 seems to be overfitting. Low looking RECO results
        trend_window = 100 #95 94 72 73
        min_slope = 2e-6
    print(f"Params are min slope: {min_slope}, trend window: {trend_window}")


    # --- History buffer ---
    val_r2_history = []    


    best_gpp_model_state = None
    best_reco_model_state = None

    for epoch in range(epochs):
        gpp_model.train()
        reco_model.train()

        # Forward pass
        gpp_pred = gpp_model(X_gpp_train)
        if tram:
            gpp_pred = gpp_pred * SW_IN_RAW_train
            gpp_pred = torch.relu(gpp_pred) # pos lin that they use in the paper.

        reco_pred = reco_model(X_reco_train)
        # nee_pred = gpp_pred + reco_pred # THIS IS THE WRONG APPROACH.
        nee_pred =  reco_pred - gpp_pred

        # Training loss
        loss = criterion(nee_pred, y_train)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        gpp_model.eval()
        reco_model.eval()
        with torch.no_grad():
            # Validation predictions
            val_gpp_pred = gpp_model(X_gpp_val)
            if tram:
                # val_gpp_pred = val_gpp_pred * X_gpp_val[:, 0].unsqueeze(1)
                val_gpp_pred = val_gpp_pred * SW_IN_RAW_val
                val_gpp_pred = torch.relu(val_gpp_pred)

            val_reco_pred = reco_model(X_reco_val)
            # val_nee_pred = val_gpp_pred + val_reco_pred # THIS IS THE WRONG APPROACH.
            val_nee_pred = val_reco_pred - val_gpp_pred

            # __Move results to CPU before computing metrics__
            train_r2 = r2_score(y_train.cpu(), nee_pred.cpu())
            val_r2 = r2_score(y_val.cpu(), val_nee_pred.cpu())
            val_loss = criterion(val_nee_pred, y_val)

        ####################################
        ##### EARLY STOPPING CONDITION #####
        ####################################
        # if val_r2 > best_val_r2 + min_delta :
        #     best_val_r2 = val_r2
        #     epochs_since_improvement = 0

        #     # Preserve the best model in case training goes bad.
        #     best_gpp_model_state = gpp_model.state_dict()
        #     best_reco_model_state = reco_model.state_dict()
        # else:
        #     epochs_since_improvement += 1

        # if epochs_since_improvement >= patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     print(f"Epoch {epoch:5d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | "
        #           f"Train R²: {train_r2.item():.4f} | Val R²: {val_r2.item():.4f}")
        #     break        
        ####################################
        ### EARLY STOPPING CONDITION END ###
        ####################################



        ####################################
        ##### EARLY STOPPING CONDITION #####
        ####################################
        val_r2_history.append(val_r2)

        # Keep only the last `trend_window` values
        if len(val_r2_history) > trend_window:
            val_r2_history.pop(0)

        # Always track best model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_gpp_model_state = gpp_model.state_dict()
            best_reco_model_state = reco_model.state_dict()

        # Apply early stopping logic only when we have enough data
        if len(val_r2_history) == trend_window:
            x = list(range(trend_window))
            y = val_r2_history
            slope, _, _, _, _ = linregress(x, y)

            if slope < min_slope:
                print(f"Early stopping at epoch {epoch} due to flat/negative trend (slope={slope:.6f})")
                print(f"Epoch {epoch:5d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | "
                    f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}")
                break        
        ####################################
        ### EARLY STOPPING CONDITION END ###
        ####################################

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | "
                  f"Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}")
        # # if epoch % 500 == 0 and len(val_r2_history) == trend_window:
        # if epoch % 500 == 0 :
        #     print(f"[epoch {epoch}] trend slope: {slope:.6f}")

    if best_gpp_model_state is not None and \
        best_reco_model_state is not None: # Second condition is not needed. Just here for clarity.
        gpp_model.load_state_dict(best_gpp_model_state)
        reco_model.load_state_dict(best_reco_model_state)

    return gpp_model, reco_model, val_r2


def compute_doy_sin_cos(date_strings):
    """
    Given a 1D torch tensor of strings in format 'YYYY-MM-DD',
    return DOY_sin and DOY_cos as torch tensors.
    """
    dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in date_strings]

    # Day of year
    doy = torch.tensor([d.timetuple().tm_yday for d in dates], dtype=torch.float32)

    # Days in year (handle leap years)
    days_in_year = torch.tensor([366 if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)) else 365 for d in dates], dtype=torch.float32)

    # Compute angles
    angle = 2 * torch.pi * doy / days_in_year

    # Compute sin and cos
    doy_sin = torch.sin(angle)
    doy_cos = torch.cos(angle)

    return doy_sin, doy_cos


def convert_hhmm_to_float_hour(time_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of HHMM-style times (e.g., 30, 1330, 0) into float hours (e.g., 0.5, 13.5, 0.0).
    
    Args:
        time_tensor (torch.Tensor): shape [N, 1] or [N], containing times in HHMM format
    
    Returns:
        torch.Tensor: shape [N], times as float hours in [0, 23.5]
    """
    time_tensor = time_tensor.squeeze()  # [N, 1] → [N] if needed
    hours = torch.floor(time_tensor / 100)
    minutes = time_tensor % 100
    float_hours = hours + (minutes / 60.0)
    return float_hours.unsqueeze(dim = 1)


def load_data(
    file_path: str,
    input_features: List[str],
    target_features: List[str],
    dropna: bool = False,
    prep_doy_sin_cos: bool = True,
    return_feature_names: bool = True,
) -> Union[ # Either return the feature name lists or not.
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]
]:    
    """
    Loads the dataset, processes it, and returns input and target tensors.
    Optionally returns input and target feature names.
    By default, this function will compute DOY_sin, DOY_cos.

    Args:
        file_path (str): Path to the CSV file.
        input_features (List[str]): Features to be used as input to the model.
        target_features (List[str]): Features to be used as targets for loss calculation.
        dropna (bool): If True, drop rows with NaN in selected columns.
        prep_doy_sin_cos (bool): If True, prep the DOY sin cos vals based on DATE (Dates are formatted as YYYY-MM-DD)

    Returns:
        Tuple containing:
            - input_tensor (torch.Tensor)
            - target_tensor (torch.Tensor)
            - (optional) input_features (List[str])
            - (optional) target_features (List[str])        
    """
    # Load the CSV
    df = pd.read_csv(file_path)
    
    if prep_doy_sin_cos:
        df["DOY_sin"], df["DOY_cos"] = compute_doy_sin_cos(df["DATE"])
        # print(df["DOY_sin"], df["DOY_cos"])


    # Select only the required columns
    data = df[input_features + target_features]

    # Split into input and target
    inputs = data[input_features].astype(float).values
    targets = data[target_features].astype(float).values

    # Convert to PyTorch tensors
    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)

    if return_feature_names:
        return input_tensor, target_tensor, input_features, target_features
    else:
        return input_tensor, target_tensor


def pair_plotter(data):
    # Plots two variables on the same graph.
    # Takes in a torch tensor of shape n x 2 (where n is num data points)
    # example: data = your_tensor  # Shape: [87647, 2]
    measurement1 = data[:, 0].numpy()
    measurement2 = data[:, 1].numpy()

    plt.figure(figsize=(15, 5))
    # to see as individual lines
    # plt.plot(measurement1, label='Measurement 1', color='blue', linewidth=1,  alpha=0.8)
    # plt.plot(measurement2, label='Measurement 2', color='orange', linewidth=1,  alpha=0.8)
    plt.scatter(range(len(measurement1)), measurement1, label='Measurement 1', color='blue', s=10, alpha=0.8)
    plt.scatter(range(len(measurement2)), measurement2, label='Measurement 2', color='orange', s=10, alpha=0.8)

    
    plt.legend()
    plt.title("Time-Series of Measurements")
    plt.xlabel("Time")
    plt.ylabel("Measurement Value")
    plt.grid(True)
    plt.show()


    # for comparing data distribution
    plt.hist(measurement1, bins=100, alpha=0.5, label='Measurement 1')
    plt.hist(measurement2, bins=100, alpha=0.5, label='Measurement 2')

    plt.legend()
    plt.title("Histogram of Measurements")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def quad_plotter(data) :
    measurement1 = data[:, 0].numpy()
    measurement2 = data[:, 1].numpy()
    measurement3 = data[:, 2].numpy()
    measurement4 = data[:, 3].numpy()

    plt.figure(figsize=(15, 5))

    # Line plots for each measurement
    # plt.plot(measurement1, label='Measurement 1', color='blue', linewidth=1, alpha=0.8)
    # plt.plot(measurement2, label='Measurement 2', color='orange', linewidth=1, alpha=0.8)
    # plt.plot(measurement3, label='Measurement 3', color='green', linewidth=1, alpha=0.8)
    # plt.plot(measurement4, label='Measurement 4', color='red', linewidth=1, alpha=0.8)

    plt.scatter(range(len(measurement1)), measurement1, label='Measurement 1', color='blue', s=10, alpha=0.8)
    plt.scatter(range(len(measurement2)), measurement2, label='Measurement 2', color='orange', s=10, alpha=0.8)    
    plt.scatter(range(len(measurement3)), measurement3, label='Measurement 3', color='green', s=10, alpha=0.8)
    plt.scatter(range(len(measurement4)), measurement4, label='Measurement 4', color='red', s=10, alpha=0.8)    

    plt.legend()
    plt.title("Line Plot of Measurements")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Histogram comparison
    plt.figure(figsize=(15, 5))
    plt.hist(measurement1, bins=100, alpha=0.5, label='Measurement 1', color='blue')
    plt.hist(measurement2, bins=100, alpha=0.5, label='Measurement 2', color='orange')
    plt.hist(measurement3, bins=100, alpha=0.5, label='Measurement 3', color='green')
    plt.hist(measurement4, bins=100, alpha=0.5, label='Measurement 4', color='red')

    plt.legend()
    plt.title("Histogram of Measurements")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def block_average_and_diff_expand(data: torch.Tensor, block_size: int):
    """
    Computes:
    1. Half-hourly diffs (data[t+1] - data[t]) with 0 prepended.
    2. Daily block averages, repeated to original shape.
    3. Daily differences between block means, repeated to original shape.

    Args:
        data (torch.Tensor): Input tensor of shape [N, D]
        block_size (int): Block size (e.g., 48 for half-hourly data to get daily stats, or 24 for hourly data)

    Returns:
        half_hourly_diff (torch.Tensor): [N, D] — difference between each time point and the previous
        daily_avg (torch.Tensor): [N, D] — repeated daily average per block
        daily_diff (torch.Tensor): [N, D] — repeated daily difference per block
    """
    
    N, D = data.shape
    num_full_blocks = N // block_size
    remainder = N % block_size

    # === Half-hourly difference ===
    half_hourly_diff = torch.zeros_like(data)
    half_hourly_diff[1:] = data[1:] - data[:-1]
    

    # === Get full blocks (i.e of block size 48) ===
    full_data = data[:num_full_blocks * block_size]
    reshaped = full_data.view(num_full_blocks, block_size, D)
    
    # === Daily averages ===
    block_means = reshaped.mean(dim=1)  # [num_blocks, D]
    block_diffs = torch.zeros_like(block_means)
    # === Daily differences ===
    block_diffs[1:] = block_means[1:] - block_means[:-1]   # [num_blocks, D]

    # === Expand to full part ===
    avg_expanded = block_means.unsqueeze(1).expand(-1, block_size, -1).reshape(-1, D)
    diff_expanded = block_diffs.unsqueeze(1).expand(-1, block_size, -1).reshape(-1, D)

    # === Handle trailing part === 
    # === Not mandatory. Just needed to ensure similar data set size for each feature ===
    if remainder > 0:
        # Use last full block's mean and diff for trailing entries
        last_avg = block_means[-1].unsqueeze(0).expand(remainder, -1)
        last_diff = block_diffs[-1].unsqueeze(0).expand(remainder, -1)

        avg_expanded = torch.cat([avg_expanded, last_avg], dim=0)
        diff_expanded = torch.cat([diff_expanded, last_diff], dim=0)

    # Final check
    assert avg_expanded.shape == data.shape
    assert diff_expanded.shape == data.shape

    return half_hourly_diff, avg_expanded, diff_expanded


def compute_gpp_prox_and_nightly_nee_avg (sw_in, nee, block_size: int = 48):
    """
    Computes GPP_prox from half-hourly SW_IN and NEE data.
        GPP_PROX = (NEEDAY − NEENIGHT ) ∗ k 
        where DAY is defined as timestamps where SW_IN > 10 W/m²
        and k is the fraction of daytime hours for each day.
    
    Args:
        SW_IN: The SW_IN, [N, 1], with N divisible by block_size
        NEE: [N, 1]
        block_size (int): Block size (e.g., 48 for half-hourly data to get daily stats)
    

    Returns GPP_prox as a 1D tensor of shape [D] (one per day).
    """

    assert sw_in.shape == nee.shape, "SW_IN and NEE must have the same shape"
    assert sw_in.shape[0] % block_size == 0, f"Number of data rows must be multiple of block_size={block_size}"

    # Reshape to [days, block_size]
    D = sw_in.shape[0] // block_size
    sw_in = sw_in.view(D, block_size)
    nee = nee.view(D, block_size)

    # Create day hour/night hour masks
    is_day = sw_in > 10
    is_night = ~is_day

    # Count how many half-hourly points are daytime (to compute k)
    day_counts = is_day.sum(dim=1).float()  # shape [D]
    k = day_counts / block_size  # shape [D]

    # Avoid division by zero (in case of zero day counts)
    day_counts = day_counts.masked_fill(day_counts == 0, 1.0)
    night_counts = is_night.sum(dim=1).float().masked_fill(is_night.sum(dim=1) == 0, 1.0)

    # Compute daily averages
    nee_day_avg = (nee * is_day).sum(dim=1) / day_counts
    nee_night_avg = (nee * is_night).sum(dim=1) / night_counts  # shape [D]

    # Final GPP_prox
    gpp_prox_daily = (nee_day_avg - nee_night_avg) * k  # shape [D]
    
    gpp_prox_full = gpp_prox_daily.repeat_interleave(block_size)    # shape [N]
    nee_night_full = nee_night_avg.repeat_interleave(block_size)    # shape [N]
    
    gpp_and_nee = torch.stack((gpp_prox_full, nee_night_full),1)
    return gpp_and_nee


def wind_direction_to_cos_sin (wind_deg: torch.Tensor) -> torch.Tensor:
    """
    Converts wind direction in degrees to a 2D unit vector (cos, sin) representation.

    Args:
        wind_deg (torch.Tensor): Wind directions in degrees, shape [N]

    Returns:
        torch.Tensor: Tensor of shape [N, 2], where [:,0] is cos(deg), [:,1] is sin(deg)
    """
    # Convert degrees to radians
    wind_rad = wind_deg * math.pi / 180.0

    # Compute cosine and sine
    cos_vals = torch.cos(wind_rad)
    sin_vals = torch.sin(wind_rad)

    # Combine into a single tensor
    wind_vec = torch.cat((cos_vals, sin_vals),1)

    return wind_vec


def normalize_features(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalizes each feature in X independently using:
        X_norm = 2 * ((X - X_min) / (X_max - X_min) - 0.5)

    Args:
        X (torch.Tensor): Input tensor of shape [N, D]

    Returns:
        X_norm (torch.Tensor): Normalized tensor of shape [N, D]
        X_min (torch.Tensor): Minimum values per feature [D]
        X_max (torch.Tensor): Maximum values per feature [D]
    """
    X_min = X.min(dim=0).values
    X_max = X.max(dim=0).values

    # Prevent division by zero
    range_ = (X_max - X_min).clamp(min=1e-8)

    X_norm = 2 * ((X - X_min) / range_ - 0.5)

    return X_norm, X_min, X_max

def unnormalize_features(X_norm: torch.Tensor, X_min: torch.Tensor, X_max: torch.Tensor) -> torch.Tensor:
    """
    Un-normalizes a normalized tensor using:
        X = ((X_norm / 2) + 0.5) * (X_max - X_min) + X_min

    Args:
        X_norm (torch.Tensor): Normalized tensor of shape [N, D]
        X_min (torch.Tensor): Minimums per feature [D]
        X_max (torch.Tensor): Maximums per feature [D]

    Returns:
        X (torch.Tensor): Un-normalized tensor of shape [N, D]
    """
    # Make sure all data is on the same device before any computation:
    device = X_norm.device
    X_min = X_min.to(device)
    X_max = X_max.to(device)

    range_ = (X_max - X_min).clamp(min=1e-8)
    return ((X_norm / 2) + 0.5) * range_ + X_min


# TODO: there is a bug! when loading reco_input_features and reco_target_features, if you have the same variable (SW_IN_1_1_1)
# the same data will be pulled two times to both tensors.

def prepare_data_using_csv (file_name, block_size):
    all_feature_names = []
    # TODO: Variable-ize the gpp features and the reco features for ease of use
    # ALL NECESSARY GPP FEATURES ["SW_IN", "PotRad", "VPD", "TA", "TS_1", "TS_2", "TS_3", "TS_4", "WTD", "WS", "WD"]
    # ALL NECESSARY RECO FEATURES = ["TA", "TS_1", "TS_2", "TS_3", "TS_4", "WTD", "WS", "WD"]
    # ALL NECESSARY OUTPUT FEATURES = ["NEE"]
    
    # 1. Calculate Day of Year Cos/Sine
    doy_cos_sin, _, feature_name, _ = load_data("data/{}".format(file_name), ["DOY_sin", "DOY_cos"] , [])
    all_feature_names.extend(feature_name)

    # 1.5 Ensure that all we have a multiple of 48 (ie. every single day is covered completely.)
    assert doy_cos_sin.shape[0] % block_size == 0, f"Number of data rows must be multiple of block_size={block_size}"

    # 2. Just import the time of day variable. Currently, will only be used for visualization.
    time_hhmm, _, feature_name, _ = load_data("data/{}".format(file_name), ["TIME"] , [])
    time_float = convert_hhmm_to_float_hour(time_hhmm)
    all_feature_names.extend(feature_name)

    # 3. Collect all variables that haven't been normalized yet. 
    #    DT_GPP,NT_GPP,DT_RECO,NT_RECO will not be normalized as they are only used for metric calculation purposes
    #    The remaining raw features will be normalized later, once the gaps have been dealt with.
    measured_features_raw = ["NEE", "SW_IN", "VPD", "TA", "TS_1", "TS_2", "TS_3", "TS_4", "WS", "DT_GPP", "NT_GPP", "DT_RECO", "NT_RECO", "Salinity"]
    measured_features_tensor_raw, _, feature_name, _ = load_data("data/{}".format(file_name), measured_features_raw, [])
    all_feature_names.extend(feature_name)


    # 4.1 Prep the WTD, WTD half hourly diff, WTD daily avg, WTD, daily diff.
    wtd, _, feature_name, _ = load_data("data/{}".format(file_name), ["WTD"], [])
    half_hourly_diff_wtd, daily_avg_wtd, daily_diff_wtd = block_average_and_diff_expand(wtd, block_size)
    all_wtd_data = torch.cat((wtd, half_hourly_diff_wtd, daily_avg_wtd, daily_diff_wtd), 1)

    all_feature_names.extend(feature_name)
    all_feature_names.extend(["WTD_HalfHourlyDiff", "WTD_DailyAvg", "WTD_DailyDiff"])

    # 4.2 Prep the pot radiation half hourly diff, daily average, and daily average diff
    pot_rad_half_hourly, _, feature_name, _ = load_data("data/{}".format(file_name), ["PotRad"], [])
    half_hourly_diff, daily_avg, daily_diff = block_average_and_diff_expand(pot_rad_half_hourly, block_size)
    all_pot_rad_data = torch.cat((pot_rad_half_hourly, half_hourly_diff, daily_avg, daily_diff), 1)
    
    all_feature_names.extend(feature_name)
    all_feature_names.extend(["PotRadHalfHourlyDiff", "PotRadDailyAvg", "PotRadDailyDiff"])





    # 5. For wind direction, convert degrees (0 to 360) into sin/cos representation
    wd, _, feature_name, _ = load_data("data/{}".format(file_name), ["WD"], []);
    wd_cos_sin = wind_direction_to_cos_sin(wd) # returns an N x 2 torch tensor. Column [:,0] is cos, [:,1] is sin representation.
    all_feature_names.extend(["WD_COS", "WD_SIN"])

    
    # 6. Calculate GPP_prox, and nightly NEE average using SW_IN and the NEE.
    #    The values we receive are NOT NORMALIZED, and later-in-the-pipeline will be normalized.  
    sw_in, _, feature_name, _ = load_data("data/{}".format(file_name), ["SW_IN"], [])
    nee, _, feature_name, _ = load_data("data/{}".format(file_name), ["NEE"], [])
    gpp_prox_and_nightly_nee_average = compute_gpp_prox_and_nightly_nee_avg(sw_in, nee)
    all_feature_names.extend(["GPP_PROX", "NIGHTLY_NEE_AVG"])

    return torch.cat((doy_cos_sin, time_float, measured_features_tensor_raw, all_wtd_data, \
                      all_pot_rad_data, wd_cos_sin, gpp_prox_and_nightly_nee_average), 1), \
                        all_feature_names
    

def load_and_clean_csv(file_path: str, block_size: int, drop_value: float = -9999.0) -> pd.DataFrame:
    """
    Loads a CSV file, drops rows where any value equals `drop_value`.

    Args:
        file_path (str): Path to the CSV file
        drop_value (float): Value to treat as missing (default: -9999)

    Returns:
        pd.DataFrame: Cleaned DataFrame with no -9999s
    """
    df = pd.read_csv(file_path)

    # Count how many times drop_value appears in each column
    missing_counts = (df == drop_value).sum()

    # Calculate missing ratio as percentage
    missing_ratio = (missing_counts / len(df)) * 100
    print(f"Ratios of missing variables in file {file_path}:\n{missing_ratio.sort_values(ascending=False)}")

    # Identify rows with any drop_value
    drop_mask = (df == drop_value).any(axis=1)

    # Extend mask to include 48*3 rows above and below. (48 is the data per day. And *3 is because a missing data
    #   point can dirty up the day before and day after. So drop off 3 days of data around the missing data points.
    # This is VERY IMPORTANT. OTHERWISE HALF HOURLY DIFF/ daily diff VARIABLES MIGHT HAVE VALUES LIKE 9997.531.
    extended_mask = drop_mask.copy()
    extended_mask |= drop_mask.shift(block_size*3, fill_value=False)
    extended_mask |= drop_mask.shift(-block_size*3, fill_value=False)

    # Drop the marked rows and reset index
    clean_df = df[~extended_mask].reset_index(drop=True)

    return clean_df

def split_data(gpp_inputs, reco_inputs, true_nee, time, sw_in_raw, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    assert gpp_inputs.shape[0] == reco_inputs.shape[0] == true_nee.shape[0] == time.shape[0] == sw_in_raw.shape[0], "Inputs must have same number of rows"

    N = gpp_inputs.shape[0]
    torch.manual_seed(seed)
    
    # Shuffle indices
    indices = torch.randperm(N)

    # Compute split sizes
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val

    # Split indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Return split tensors
    return {
        'train': {
            'gpp': gpp_inputs[train_idx],
            'reco': reco_inputs[train_idx],
            'nee': true_nee[train_idx],
            'sw_in_raw': sw_in_raw[train_idx],
            'time': time[train_idx]
        },
        'val': {
            'gpp': gpp_inputs[val_idx],
            'reco': reco_inputs[val_idx],
            'nee': true_nee[val_idx],
            'sw_in_raw': sw_in_raw[val_idx],
            'time': time[val_idx]
        },
        'test': {
            'gpp': gpp_inputs[test_idx],
            'reco': reco_inputs[test_idx],
            'nee': true_nee[test_idx],
            'sw_in_raw': sw_in_raw[test_idx],
            'time': time[test_idx]
        }
    }

def get_first_5_letters(filename):
    """
    Given a file name, returns the site acronym.
    Example
        Input: CADSM_nee_partition_202101010000_202512312359.csv
        Output: CADSM
    """
    match = re.search(r'[a-zA-Z]{5}', filename)
    return match.group(0) if match else None

def load_trained_model(trained_gpp_model, trained_reco_model, site_name, run_type_str, device):
    """
    Loads the already trained model.
    Example:
    Inputs:
        site_name = "CADSM"
        run_type_str = "Tramontana" (or "Custom")
    Outputs:
        The trained GPP and RECO models are returned.
    """
    # "TODO the following logic to be discontinued if it works on both cpu and cuda"
    # if torch.cuda.is_available():
    #     trained_gpp_model.load_state_dict(torch.load(f"trained_models/{site_name}_gpp_model_{run_type_str}.pth", weights_only=True))
    #     trained_reco_model.load_state_dict(torch.load(f"trained_models/{site_name}_reco_model_{run_type_str}.pth", weights_only=True))
    # else: # map_location=torch.device('cpu') is needed for graphing on the CPU.
    #     trained_gpp_model.load_state_dict(torch.load(f"trained_models/{site_name}_gpp_model_{run_type_str}.pth", weights_only=True, map_location=torch.device('cpu')))
    #     trained_reco_model.load_state_dict(torch.load(f"trained_models/{site_name}_reco_model_{run_type_str}.pth", weights_only=True, map_location=torch.device('cpu')))

    trained_gpp_model.load_state_dict(torch.load(f"trained_models/{site_name}_gpp_model_{run_type_str}.pth",
                                                 weights_only=True, map_location=device))
    trained_reco_model.load_state_dict(torch.load(f"trained_models/{site_name}_reco_model_{run_type_str}.pth",
                                                  weights_only=True, map_location=device))

    # Then make sure to move the loaded models to device.
    # Both models, and the data has to be on the same device. Avoid any splits at all times.
    trained_gpp_model = trained_gpp_model.to(device)
    trained_reco_model = trained_reco_model.to(device)


def initialize_model(tramontana_run):
    """
    Just initialize GPP and RECO models. (with no-trained weights)
    These initialized models will be used for loading in the trained model.
    """
    if tramontana_run:
        trained_gpp_model = SNN_GPP_Tram(splits['train']['gpp'].shape[1], hidden_size)
        trained_reco_model = SNN_GPP_Tram(splits['train']['reco'].shape[1], hidden_size)
    else:
        trained_gpp_model = SNN_GPP(splits['train']['gpp'].shape[1], hidden_size)
        trained_reco_model = SNN_RECO(splits['train']['reco'].shape[1], hidden_size)
    return trained_gpp_model, trained_reco_model

def get_run_type_str(tramontana_run):
    """
    This str is useful for saving, loading, and graphing purposes.
    """
    return "Tramontana" if tramontana_run else "Custom"

def evaluate_single_model (gpp_inputs, reco_inputs, time, sw_in_raw, model_inputs_information, save_plot, plot_saving_str, results_dict, experiment_id, GPP_INPUT_FEATURES, RECO_INPUT_FEATURES):
    """
    Evaluate a single model against the DAYTIME AND NIGHTTIME MODELS
    """
    # Init model variables to prep for loading.
    trained_gpp_model, trained_reco_model = initialize_model(tramontana_run=tramontana_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Unpickling models")
    # Load models.
    load_trained_model(trained_gpp_model, trained_reco_model, site_name, run_type_str, device)
    print(f"Following models have been loaded and moved to {device}:\n"
          f"trained gpp model:\n {trained_gpp_model}, trained_reco_model:\n {trained_reco_model}")


    # Time for eval.
    trained_gpp_model.eval()
    trained_reco_model.eval()

    # test_or_train = "train"
    # test_or_train = "test"
    # test_or_train = "val"
    test_or_train = "full"
    assert test_or_train == "full" or test_or_train == "test" or test_or_train == "train" or test_or_train == "val", "Pick a split, or full dataset."

    if test_or_train == "full":
        # do nothing
        print("using full data set for visualization")
    else: # grab the corresponding splits. Whether it is train/test.
        time = splits[test_or_train]['time']
        gpp_inputs = splits[test_or_train]['gpp']
        reco_inputs = splits[test_or_train]['reco']
        sw_in_raw = splits[test_or_train]['sw_in_raw']


    with torch.no_grad():
        # with full time is just the time var.
        gpp_pred = trained_gpp_model(gpp_inputs.to(device))
        reco_pred = trained_reco_model(reco_inputs.to(device))
        if tramontana_run:
            SW_IN_RAW_values = sw_in_raw.to(device)
            gpp_pred = gpp_pred * SW_IN_RAW_values
            gpp_pred = torch.relu(gpp_pred)



    # The network is trained on the normalized NEE values. That means that the subnetwork predictions are also normalized.
    # So before plotting them, the values need to be un-normalized. For this we need the Raw NEE values from the clean file(clean_file_name)
    raw_nee_name =  ['NEE']
    # WHEN NORMALIZING RAW VALUES, IF YOUR DATA SET ALREADY HAS THE DOY_SIN AND DOY_COS, YOU WANT TO SET prep_doy_sin_cos TO FALSE.
    raw_nee, _, _, _ = load_data("{}".format(clean_file_name), raw_nee_name, [], prep_doy_sin_cos = False);

    # Normalize Raw Features just to get the nee_min and nee_max vals.
    _, nee_min, nee_max = normalize_features(raw_nee)
    print(f'Un-normalizing the GPP and RECO predictions using NEE min: {nee_min}, NEE max: {nee_max}')

    reco_pred_raw = unnormalize_features(reco_pred, nee_min, nee_max)
    gpp_pred_raw = unnormalize_features(gpp_pred, nee_min, nee_max)
    # gpp_pred_raw = -gpp_pred_raw # just to flip the view #ignore for now


    df = pd.DataFrame({
        'hour': time.squeeze().cpu().numpy(),
        'gpp': gpp_pred_raw.squeeze().cpu().numpy(),
        'reco': reco_pred_raw.squeeze().cpu().numpy(),
    })


    mean_or_median = 'mean'
    # Group and compute mean ± std
    gpp_stats = df.groupby('hour')['gpp'].agg([mean_or_median, 'std'])
    reco_stats = df.groupby('hour')['reco'].agg([mean_or_median, 'std'])

    # fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig, ax = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
    fig.suptitle(f'{run_type_str} - GPP and RECO predictions with the {test_or_train} split', fontsize=16, fontweight='bold')


    # GPP plot
    ax[0].plot(gpp_stats.index, gpp_stats[mean_or_median], label=f'GPP {mean_or_median}')
    ax[0].fill_between(gpp_stats.index,
                    gpp_stats[mean_or_median] - gpp_stats['std'],
                    gpp_stats[mean_or_median] + gpp_stats['std'],
                    alpha=0.3, label='±1 Std Dev')
    ax[0].set_ylabel("GPP")
    ax[0].legend()
    ax[0].grid(True)

    # RECO plot
    ax[1].plot(reco_stats.index, reco_stats[mean_or_median], label=f'RECO {mean_or_median}', color='green')
    ax[1].fill_between(reco_stats.index,
                    reco_stats[mean_or_median] - reco_stats['std'],
                    reco_stats[mean_or_median] + reco_stats['std'],
                    alpha=0.3, label='±1 Std Dev', color='green')
    ax[1].set_xlabel("Hour of Day")
    ax[1].set_ylabel("RECO")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    if save_plot:
        plt.savefig(plot_saving_str, dpi=300)  # Save the figure with high resolution
        plt.close()
    else:
        plt.show()

    run_metrics = True
    if run_metrics:

        # reco_pred_raw and gpp_pred_raw has the NN predicted raw values
        DT_GPP = ['DT_GPP']
        dt_gpp, _, DT_GPP_name, _ = load_data("{}".format(normalized_file_name), DT_GPP, [], prep_doy_sin_cos = False)
        NT_GPP = ['NT_GPP']
        nt_gpp, _, NT_GPP_name, _ = load_data("{}".format(normalized_file_name), NT_GPP, [], prep_doy_sin_cos = False)
        DT_RECO = ['DT_RECO']
        dt_reco, _, DT_RECO_name, _ = load_data("{}".format(normalized_file_name), DT_RECO, [], prep_doy_sin_cos = False)
        NT_RECO = ['NT_RECO']
        nt_reco, _, NT_RECO_name, _ = load_data("{}".format(normalized_file_name), NT_RECO, [], prep_doy_sin_cos = False)

        print(model_inputs_information)

        # Step 1: Compute metrics
        metrics = {
            'DT_GPP_vs_model': {
                'r2': round(r2_score(dt_gpp, gpp_pred_raw), 2),
                'rmse': round(rmse(dt_gpp, gpp_pred_raw), 2)
            },
            'NT_GPP_vs_model': {
                'r2': round(r2_score(nt_gpp, gpp_pred_raw), 2),
                'rmse': round(rmse(nt_gpp, gpp_pred_raw), 2)
            },
            'DT_RECO_vs_model': {
                'r2': round(r2_score(dt_reco, reco_pred_raw), 2),
                'rmse': round(rmse(dt_reco, reco_pred_raw), 2)
            },
            'NT_RECO_vs_model': {
                'r2': round(r2_score(nt_reco, reco_pred_raw), 2),
                'rmse': round(rmse(nt_reco, reco_pred_raw), 2)
            },
            'DT_GPP_vs_NT_GPP': {
                'r2': round(r2_score(dt_gpp, nt_gpp), 2),
                'rmse': round(rmse(dt_gpp, nt_gpp), 2)
            },
            'DT_RECO_vs_NT_RECO': {
                'r2': round(r2_score(dt_reco, nt_reco), 2),
                'rmse': round(rmse(dt_reco, nt_reco))
            }
        }

        # Step 2: Save metrics into master dictionary
        results_dict[f"experiment_{experiment_id}"] = {
            'run_type': run_type_str,
            'gpp_inputs': GPP_INPUT_FEATURES,
            'reco_inputs': RECO_INPUT_FEATURES,
            'metrics': metrics
        }

        # Step 3: Print metrics for logging.
        print(
            f"DT_GPP vs {run_type_str}_GPP R²={metrics['DT_GPP_vs_model']['r2']:.2f} RMSE={metrics['DT_GPP_vs_model']['rmse']:.2f}\n"
            f"NT_GPP vs {run_type_str}_GPP R²={metrics['NT_GPP_vs_model']['r2']:.2f} RMSE={metrics['NT_GPP_vs_model']['rmse']:.2f}\n"
            f"DT_RECO vs {run_type_str}_RECO R²={metrics['DT_RECO_vs_model']['r2']:.2f} RMSE={metrics['DT_RECO_vs_model']['rmse']:.2f}\n"
            f"NT_RECO vs {run_type_str}_RECO R²={metrics['NT_RECO_vs_model']['r2']:.2f} RMSE={metrics['NT_RECO_vs_model']['rmse']:.2f}\n"
            f"DT_GPP vs NT_GPP R²={metrics['DT_GPP_vs_NT_GPP']['r2']:.2f} RMSE={metrics['DT_GPP_vs_NT_GPP']['rmse']:.2f}\n"
            f"DT_RECO vs NT_RECO R²={metrics['DT_RECO_vs_NT_RECO']['r2']:.2f} RMSE={metrics['DT_RECO_vs_NT_RECO']['rmse']:.2f}\n"
        )



pre_processing = False
drop_na = False
normalize_raw_features = False
run_experiments = True
train_models = True # if you don't train, the existing model will be loaded for evaluation.
save_models = True # you can train to see the results. But you don't have to save the model.
hidden_size = 12

##############################################
#### Use Tramontana model or Custom Model ####
##############################################
tramontana_run = False
run_type_str = get_run_type_str(tramontana_run=tramontana_run)

print(f"pre_processing: {pre_processing}\
      drop_na: {drop_na} \
      normalize_raw_features: {normalize_raw_features} \
      train_models: {train_models} \
      save_models: {save_models} \
      hidden_size: {hidden_size} \
      tramontana_run: {tramontana_run} \
      ")

# file_name = "CADSM_nee_partition_202101010000_202512312359.csv"
file_name = "CADSM_nee_partition_202109170000_202505292359.csv"
# site_name = get_first_5_letters(filename=file_name)
site_name = "temp"
if site_name is None:
    raise Exception(f"\n\n\nTried getting the site name using the file_name variable but failed.\n"
                    "Make sure you specified a file name using the variable file_name.")


processed_file_name = "data/Processed_{}".format(file_name)
clean_file_name = "data/Cleaned_{}".format(file_name) # Will hold the rows that doesn't have NaN values

# ONLY NORMALIZE THE CLEAN FILE. Otherwise -9999's will affect the normalization.
normalized_file_name = "data/Normalized_{}".format(file_name)
block_size = 48 #half hourly data leads to 48 data points per day.

if pre_processing:
    print("pre-processing the third stage file to obtain/calculate the necessary features")
    try:
        prepped_data, feature_names = prepare_data_using_csv(file_name, block_size)
    except ValueError:
        raise Exception(f"\n\n\n\n\nEnsure that the {file_name} file doesn't have units! \
                        \nIt should only have the feature names, and the corresponding values.\n\n\n\n\n")
    except Exception as e:
        raise e

    print("prepped data shape was", prepped_data.shape)
    print("saving the following features", feature_names)

    X_df = pd.DataFrame(prepped_data.numpy(), columns=feature_names)
    X_df.to_csv(processed_file_name, index=False)



if drop_na:
    # load_and_clean will drop all rows at least one missing value (ie. -9999)
    print("dropping rows with missing values")
    df_clean = load_and_clean_csv(processed_file_name, block_size)

    original_file_len = len(pd.read_csv(processed_file_name))
    cleaned_file_len = len(df_clean)
    print(f"Original rows: {original_file_len}")
    print(f"Cleaned rows:  {cleaned_file_len}")
    print(f"Preserved data ratio: {cleaned_file_len/original_file_len}")
    """
    Original rows: 87600
    Cleaned rows:  56870
    Ratio: 56870/87600 = %~64.9 preserved! (and even more considering I included data till the end of 2025)
    Let me calculate the true preservation rate because it is actually relevant and important.
    bc my orig csv has data till 2025-12-31 (ie empty)
    and the orig has data till 2025-05-31. Need to delete a lot of data points.
    so for CADSM- delete everything after row 77338.
    Then run load_and_clean_csv. Result:
    Original rows: 77336
    Cleaned rows:  56870
    Ratio: %~73.5 preserved.
    """

    df_clean.to_csv(clean_file_name, index=False)


# then take the clean file, and normalize all that has to be normalized.
if normalize_raw_features:
    # all feature_names ['DOY_sin', 'DOY_cos', 'NEE', 'SW_IN', 'VPD', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'WD_COS', 'WD_SIN', 'GPP_PROX', 'NIGHTLY_NEE_AVG', 'Salinity', 'WTD', 'WTD_HalfHourlyDiff', 'WTD_DailyAvg', 'WTD_DailyDiff']
    
    raw_feature_names =  ['NEE', 'SW_IN', 'VPD', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'GPP_PROX', 'NIGHTLY_NEE_AVG', 'Salinity', 'WTD', 'WTD_HalfHourlyDiff', 'WTD_DailyAvg', 'WTD_DailyDiff']
    # WHEN NORMALIZING RAW VALUES, IF YOUR DATA SET ALREADY HAS THE DOY_SIN AND DOY_COS, YOU WANT TO SET prep_doy_sin_cos TO FALSE.
    raw_features, _, raw_feature_name, _ = load_data("{}".format(clean_file_name), raw_feature_names, [], prep_doy_sin_cos = False);
    
    # Normalize Raw Features:
    normalized_raw_features, _, _ = normalize_features(raw_features)

    # 'DOY_sin', 'DOY_cos', 'TIME', 'WD_COS', 'WD_SIN' do not need to be normalized as the are already normalized.
    # 'DT_GPP', 'NT_GPP', 'DT_RECO', 'NT_RECO' will only be used for metrics. We don't need to normalize them.
    other_feature_names = ['DOY_sin', 'DOY_cos', 'TIME', 'WD_COS', 'WD_SIN', 'DT_GPP', 'NT_GPP', 'DT_RECO', 'NT_RECO']
    other_features, _, other_feature_names, _ = \
            load_data("{}".format(clean_file_name), other_feature_names, [], prep_doy_sin_cos = False);

    
    all_feature_names = []
    all_feature_names.extend(other_feature_names)
    all_feature_names.extend(raw_feature_names)

    all_features = torch.cat((other_features, normalized_raw_features), 1)

    df_normalized = pd.DataFrame(all_features.numpy(), columns=all_feature_names)
    df_normalized.to_csv(normalized_file_name, index=False)    

if not run_experiments:
    sys.exit("Stopping before running the experiments.")
# Read normalized values file then do backprop magic time.

# OG structure:
# GPP_INPUT_FEATURES = ['SW_IN', 'VPD', 'TA', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'WD_COS', 'WD_SIN', 'GPP_PROX']
# RECO_INPUT_FEATURES = ['DOY_sin', 'DOY_cos', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'WD_COS', 'WD_SIN', 'NIGHTLY_NEE_AVG']
NEE = ['NEE']
TIME = ['TIME']

GPP_INPUT_FEATURES_SETS = [
    ['SW_IN', 'TA'],
    ['SW_IN', 'TA', 'VPD'],
    ['SW_IN', 'TA', 'VPD', 'WS'],
    ['SW_IN', 'TA', 'VPD', 'WS', 'WD_COS', 'WD_SIN'], # 4
    ['SW_IN', 'TA', 'VPD', 'WS', 'WD_COS', 'WD_SIN', 'WTD'], # 5
    ['SW_IN', 'TA', 'VPD', 'WS', 'WD_COS', 'WD_SIN', 'WTD', 'Salinity'], # 6
    ['SW_IN', 'TA', 'VPD', 'WS', 'WD_COS', 'WD_SIN', 'WTD', 'Salinity', 'WTD_HalfHourlyDiff'], # 7
    ['SW_IN', 'TA', 'VPD', 'WS', 'WD_COS', 'WD_SIN', 'WTD', 'Salinity', 'WTD_HalfHourlyDiff', 'WTD_DailyAvg', 'WTD_DailyDiff'], # 8

    # DAILY VARS
    ['PotRadDailyAvg', 'PotRadDailyDiff', 'GPP_PROX'], # 9
    # Non-daily vars
    ['SW_IN', 'VPD', 'TA', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'WD_COS', 'WD_SIN'], # 10
    # Full Vars
    ['SW_IN', 'VPD', 'TA', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'WD_COS', 'WD_SIN', 'GPP_PROX'], # 11
    # Full Vars (tidal diff and avg) NO DAILY TIDAL VARS
    ['SW_IN', 'VPD', 'TA', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'WD_COS', 'WD_SIN', 'GPP_PROX', 'Salinity', 'WTD_HalfHourlyDiff'], # 12
    # Full Vars (tidal diff and avg)
    ['SW_IN', 'VPD', 'TA', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'WD_COS', 'WD_SIN', 'GPP_PROX', 'Salinity', 'WTD_HalfHourlyDiff', 'WTD_DailyAvg', 'WTD_DailyDiff'], # 13

]
RECO_INPUT_FEATURES_SETS = [
    ['DOY_sin', 'DOY_cos', 'TA'],
    ['DOY_sin', 'DOY_cos', 'TA'],
    ['DOY_sin', 'DOY_cos', 'TA', 'WS'],
    ['DOY_sin', 'DOY_cos', 'TA', 'WS', 'WD_COS', 'WD_SIN'], # 4
    ['DOY_sin', 'DOY_cos', 'TA', 'WS', 'WD_COS', 'WD_SIN', 'WTD'], # 5
    ['DOY_sin', 'DOY_cos', 'TA', 'WS', 'WD_COS', 'WD_SIN', 'WTD', 'Salinity'], # 6
    ['DOY_sin', 'DOY_cos', 'TA', 'WS', 'WD_COS', 'WD_SIN', 'WTD', 'Salinity', 'WTD_HalfHourlyDiff'], # 7
    ['DOY_sin', 'DOY_cos', 'TA', 'WS', 'WD_COS', 'WD_SIN', 'WTD', 'Salinity', 'WTD_HalfHourlyDiff', 'WTD_DailyAvg', 'WTD_DailyDiff'], # 8

    # DAILY VARS
    ['DOY_sin', 'DOY_cos', 'NIGHTLY_NEE_AVG'], # 9
    # Non-daily vars
    ['TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'WD_COS', 'WD_SIN'], # 10
    # Full Vars
    ['DOY_sin', 'DOY_cos', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'WD_COS', 'WD_SIN', 'NIGHTLY_NEE_AVG'], # 11
    # Full Vars (tidal diff and avg) NO DAILY TIDAL VARS
    ['DOY_sin', 'DOY_cos', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'WD_COS', 'WD_SIN', 'NIGHTLY_NEE_AVG', 'Salinity', 'WTD_HalfHourlyDiff'], # 12
    # Full Vars (tidal diff and avg)
    ['DOY_sin', 'DOY_cos', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'WD_COS', 'WD_SIN', 'NIGHTLY_NEE_AVG', 'Salinity', 'WTD_HalfHourlyDiff', 'WTD_DailyAvg', 'WTD_DailyDiff'], # 13
]

assert len(GPP_INPUT_FEATURES_SETS) == len(RECO_INPUT_FEATURES_SETS), "You need to have the same number of subsets"

results_dict = {} # this will store all experiment outputs.
for experiment_id in range(len(GPP_INPUT_FEATURES_SETS)):
    GPP_INPUT_FEATURES = GPP_INPUT_FEATURES_SETS[experiment_id]
    RECO_INPUT_FEATURES = RECO_INPUT_FEATURES_SETS[experiment_id]

# Read all data. Both normalized variables, and the variables that do not need to be normalized are saved in this file.
    gpp_inputs, _, gpp_input_names, _ = load_data("{}".format(normalized_file_name), GPP_INPUT_FEATURES, [], prep_doy_sin_cos = False)
    reco_inputs, _, reco_input_names, _ = load_data("{}".format(normalized_file_name), RECO_INPUT_FEATURES, [], prep_doy_sin_cos = False)
    true_nee, _, true_nee_name, _ = load_data("{}".format(normalized_file_name), NEE, [], prep_doy_sin_cos = False)
    time, _, time_name, _ = load_data("{}".format(normalized_file_name), TIME, [], prep_doy_sin_cos = False)

    print(f"gpp_input_names,  {gpp_input_names} \n"
        f"reco_input_names,  {reco_input_names} \n"
        f"true_nee_name,  {true_nee_name} \n")


    # READ THE SW_IN EVEN IF IT IS NOT A TRAMONTANA RUN.
    # READING FROM THE CLEAN FILE as the raw (not-normalized) sw_in is needed for the Tramontana model.
    # CLEAN FILE AND THE NORMALIZED FILE SHOULD HAVE THE EXACT SAME ROWS FOR THIS TO WORK PROPERLY
    sw_in_raw, _, _, _ = load_data("{}".format(clean_file_name), ['SW_IN'], [], prep_doy_sin_cos = False)


    # Not sure if this is the cleanest way. But keep for now as we need to validate.
    splits = split_data(gpp_inputs, reco_inputs, true_nee, time, sw_in_raw, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)


    if train_models:
        gpp_model, reco_model, val_r2 = better_fit_gpu(
            tram=tramontana_run,
            hidden_layer_size=hidden_size,
            X_gpp_train=splits['train']['gpp'],
            X_reco_train=splits['train']['reco'],
            y_train=splits['train']['nee'],
            X_gpp_val=splits['val']['gpp'],
            X_reco_val=splits['val']['reco'],
            y_val=splits['val']['nee'],
            SW_IN_RAW_train=splits['train']['sw_in_raw'],
            SW_IN_RAW_val=splits['val']['sw_in_raw'],
        )

    if save_models:
        torch.save(gpp_model.state_dict(), f"trained_models/{site_name}_gpp_model_{run_type_str}.pth")
        torch.save(reco_model.state_dict(), f"trained_models/{site_name}_reco_model_{run_type_str}.pth")

    model_inputs_information = (f"\n\n\nMetrics for the {run_type_str}, with\n"
                f"GPP inputs: {GPP_INPUT_FEATURES}\n"
                f"RECO inputs: {RECO_INPUT_FEATURES}:")
    save_plot = False # too long to save now # TODO: fix fig str if you need to plot with long list of vars
    # prep the plot saving str
    plot_saving_str = (f"./experiment_figures/{run_type_str}_"
                       f"gpp_{'_'.join(GPP_INPUT_FEATURES)}_"
                       f"reco_{'_'.join(RECO_INPUT_FEATURES)}"
                       )

    evaluate_single_model(gpp_inputs, reco_inputs, time, sw_in_raw, model_inputs_information, save_plot, plot_saving_str, results_dict, experiment_id, GPP_INPUT_FEATURES, RECO_INPUT_FEATURES)
    results_dict[f"experiment_{experiment_id}"]['metrics']['val_r2'] = round(val_r2, 4)



print(json.dumps(results_dict, indent = 4))
for experiment_id in results_dict:
  experiment = results_dict[experiment_id]
  print(f""
        # f"{experiment_id}\n"
        # f"gpp inputs {'_'.join(experiment['gpp_inputs'])}\n"
        # f"reco inputs {'_'.join(experiment['reco_inputs'])}\n"
        f"{experiment['metrics']['val_r2']},"
        f"{experiment['metrics']['DT_GPP_vs_model']['r2']},"
        f"{experiment['metrics']['NT_GPP_vs_model']['r2']},"
        f"{experiment['metrics']['DT_RECO_vs_model']['r2']},"
        f"{experiment['metrics']['NT_RECO_vs_model']['r2']}"
  )
"""
TODO:
- DONE - more elaborate early stopping condition
- Init the device ONLY ONCE.

"""

"""
Tramontana - A LOT SLOWER OF IMPROVEMENT OVER TIME:
Early stopping at epoch 12413
    patience = 500
    min_delta = 1e-1
Epoch 12413 | Train Loss: 0.015983 | Val Loss: 0.016003 | Train R²: 0.4403 | Val R²: 0.4624
Tramontana benefits from higher patience:
    patience = 1000
    min_delta = 1e-1
Early stopping at epoch 14199
Epoch 14199 | Train Loss: 0.010570 | Val Loss: 0.010720 | Train R²: 0.6299 | Val R²: 0.6399


    patience = 500
    min_delta = 1e-2
Early stopping at epoch 23590    
DT_GPP vs Tramontana_GPP R²=0.91 RMSE=2.17
NT_GPP vs Tramontana_GPP R²=0.87 RMSE=2.50
DT_RECO vs Tramontana_RECO R²=0.34 RMSE=1.88
NT_RECO vs Tramontana_RECO R²=0.15 RMSE=1.85

    patience = 1000
    min_delta = 1e-2
Early stopping at epoch 24873
Epoch 24873 | Train Loss: 0.002007 | Val Loss: 0.002173 | Train R²: 0.9297 | Val R²: 0.9270
DT_GPP vs Tramontana_GPP R²=0.94 RMSE=1.82
NT_GPP vs Tramontana_GPP R²=0.91 RMSE=2.15
DT_RECO vs Tramontana_RECO R²=0.56 RMSE=1.55
NT_RECO vs Tramontana_RECO R²=0.46 RMSE=1.48


    patience = 500
    min_delta = 1e-3
Early stopping at epoch 30177
Epoch 30177 | Train Loss: 0.001618 | Val Loss: 0.001699 | Train R²: 0.9434 | Val R²: 0.9429    
DT_GPP vs Tramontana_GPP R²=0.95 RMSE=1.58
NT_GPP vs Tramontana_GPP R²=0.94 RMSE=1.77
DT_RECO vs Tramontana_RECO R²=0.71 RMSE=1.24
NT_RECO vs Tramontana_RECO R²=0.73 RMSE=1.05


    patience = 500
    min_delta = 1e-4
Early stopping at epoch 37211
Epoch 37211 | Train Loss: 0.001496 | Val Loss: 0.001563 | Train R²: 0.9476 | Val R²: 0.9475
DT_GPP vs Tramontana_GPP R²=0.95 RMSE=1.54
NT_GPP vs Tramontana_GPP R²=0.94 RMSE=1.67
DT_RECO vs Tramontana_RECO R²=0.75 RMSE=1.17
NT_RECO vs Tramontana_RECO R²=0.78 RMSE=0.95






CUSTOM MODELS
with
    patience = 500
    min_delta = 1e-4
    Early stopping at epoch 8003    
DT_GPP vs Custom_GPP R²=0.95 RMSE=1.62
NT_GPP vs Custom_GPP R²=0.93 RMSE=1.87
DT_RECO vs Custom_RECO R²=0.69 RMSE=1.29
NT_RECO vs Custom_RECO R²=0.65 RMSE=1.19


with
    patience = 500
    min_delta = 1e-3
    Early stopping at epoch 4866    
DT_GPP vs Custom_GPP R²=0.95 RMSE=1.60
NT_GPP vs Custom_GPP R²=0.93 RMSE=1.83
DT_RECO vs Custom_RECO R²=0.72 RMSE=1.23
NT_RECO vs Custom_RECO R²=0.72 RMSE=1.07    

with 
    patience = 500
    min_delta = 1e-2
    Early stopping at epoch 1523
Epoch  1523 | Train Loss: 0.002157 | Val Loss: 0.002255 | Train R²: 0.9245 | Val R²: 0.9243    
DT_GPP vs Custom_GPP R²=0.95 RMSE=1.67
NT_GPP vs Custom_GPP R²=0.93 RMSE=1.87
DT_RECO vs Custom_RECO R²=0.78 RMSE=1.10
NT_RECO vs Custom_RECO R²=0.84 RMSE=0.80        


with 
    patience = 500
    min_delta = 1e-1
    Early stopping at epoch 851
DT_GPP vs Custom_GPP R²=0.93 RMSE=1.89
NT_GPP vs Custom_GPP R²=0.92 RMSE=2.05
DT_RECO vs Custom_RECO R²=0.69 RMSE=1.30
NT_RECO vs Custom_RECO R²=0.78 RMSE=0.94
    


    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200)

# print(f"Parameters")
# for param in trained_gpp_model.parameters():
#     print(param)
# for param in trained_reco_model.parameters():
#     print(param)
    
"""