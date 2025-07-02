import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime
import math
# from torcheval.metrics import R2Score
# from torchmetrics.functional import r2_score


class SNN_GPP_Tram(nn.Module):
    def __init__(self, input_dim):
        super(SNN_GPP_Tram, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
            # TODO: NEED TO LATER ON MULTIPLY THE OUTPUT OF THIS WITH SW_IN, THEN PUSH IT THROUGH POSLIN.
        )

    def forward(self, x):
        return self.net(x)

class SNN_RECO_Tram(nn.Module):
    def __init__(self, input_dim):
        super(SNN_RECO_Tram, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class SNN_GPP(nn.Module):
    def __init__(self, input_dim):
        super(SNN_GPP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class SNN_RECO(nn.Module):
    def __init__(self, input_dim):
        super(SNN_RECO, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def better_fit(X_gpp_train, X_reco_train, y_train, 
        X_gpp_val, X_reco_val, y_val,
        epochs=10000, lr=1e-3):

    tram = True
    # Instantiate models
    if tram:
        gpp_model = SNN_GPP_Tram(X_gpp_train.shape[1])
        reco_model = SNN_RECO_Tram(X_reco_train.shape[1])
    else:
        gpp_model = SNN_GPP(X_gpp_train.shape[1])
        reco_model = SNN_RECO(X_reco_train.shape[1])

    # Optimizer
    optimizer = optim.Adam(list(gpp_model.parameters()) + list(reco_model.parameters()), lr=lr)

    # Loss function
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        gpp_model.train()
        reco_model.train()

        # Forward pass
        gpp_pred = gpp_model(X_gpp_train)
        if tram:
            # import pdb; pdb.set_trace()
            gpp_pred = gpp_pred * X_gpp_train[:,0].unsqueeze(1) # X_gpp_train[:,0] has the SW_IN!
            gpp_pred = torch.relu(gpp_pred) # pos lin that they use in the paper.

        reco_pred = reco_model(X_reco_train)
        nee_pred = gpp_pred + reco_pred

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
                val_gpp_pred = val_gpp_pred * X_gpp_val[:,0].unsqueeze(1)
                val_gpp_pred = torch.relu(val_gpp_pred)

            val_reco_pred = reco_model(X_reco_val)
            val_nee_pred = val_gpp_pred + val_reco_pred

            # R² and losses
            train_r2 = r2_score(y_train, nee_pred)
            val_r2 = r2_score(y_val, val_nee_pred)
            val_loss = criterion(val_nee_pred, y_val)

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} | "
                  f"Train R²: {train_r2.item():.4f} | Val R²: {val_r2.item():.4f}")

    return gpp_model, reco_model
    
    

def fit(gpp_inputs, reco_inputs, true_nee):
    # Instantiate models
    input_dim_gpp = 12  # adjust based on your actual input features
    input_dim_reco = 12

    gpp_model = SNN_GPP(input_dim_gpp)
    reco_model = SNN_RECO(input_dim_reco)

    # Optimizers (can be separate or joint)
    optimizer = optim.Adam(list(gpp_model.parameters()) + list(reco_model.parameters()), lr=1e-3)

    # Loss function
    criterion = nn.MSELoss()

    # Example training loop
    for epoch in range(10000):
        gpp_model.train()
        reco_model.train()

        # Batch of inputs (replace with your actual data loader)
        # gpp_inputs = torch.randn(32, input_dim_gpp)
        # reco_inputs = torch.randn(32, input_dim_reco)
        # true_nee = torch.randn(32, 1)  # Measured NEE

        # Forward pass
        gpp_pred = gpp_model(gpp_inputs)
        reco_pred = reco_model(reco_inputs)
        nee_pred = gpp_pred + reco_pred

        # Loss and backward  MSE WORKING
        loss = criterion(nee_pred, true_nee)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        gpp_model.eval()
        reco_model.eval()
        with torch.no_grad():
            r2 = r2_score(true_nee, nee_pred)
            print("R²:", r2.item())      


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

    # ignore for now
    # Replace -9999 with NaN
    # df.replace(-9999, pd.NA, inplace=True)
    
    if prep_doy_sin_cos:
        df["DOY_sin"], df["DOY_cos"] = compute_doy_sin_cos(df["DATE"])
        # print(df["DOY_sin"], df["DOY_cos"])


    # Select only the required columns
    data = df[input_features + target_features] 

    # # Optionally drop rows with missing values. Not tested yet.
    # if dropna:
    #     data = data.dropna()

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
    assert sw_in.shape[0] % block_size == 0, f"Length must be multiple of block_size={block_size}"

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


# TODO: there is a bug! when loading reco_input_features and reco_target_features, if you have the same variable (SW_IN_1_1_1)
# the same data will be pulled two times to both tensors.

def prepare_data_using_csv (file_name):
    all_feature_names = []
    # TODO: Variable-ize the gpp features and the reco features for ease of use
    # ALL NECESSARY GPP FEATURES ["SW_IN", "PotRad", "VPD", "TA", "TS_1", "TS_2", "TS_3", "TS_4", "WTD", "WS", "WD"]
    # ALL NECESSARY RECO FEATURES = ["TA", "TS_1", "TS_2", "TS_3", "TS_4", "WTD", "WS", "WD"]
    # ALL NECESSARY OUTPUT FEATURES = ["NEE"]
    
    # 1. Calculate Day of Year Cos/Sine
    doy_cos_sin, _, feature_name, _ = load_data("data/{}".format(file_name), ["DOY_sin", "DOY_cos"] , [])
    all_feature_names.extend(feature_name)

    # 2. Collect all variables that haven't been normalized yet. 
    #    These will be normalized later once the gaps have been dealt with.
    measured_features_raw = ["NEE", "SW_IN", "VPD", "TA", "TS_1", "TS_2", "TS_3", "TS_4", "WTD", "WS"] 
    measured_features_tensor_raw, _, feature_name, _ = load_data("data/{}".format(file_name), measured_features_raw, [])
    all_feature_names.extend(feature_name)


    # 3. Prep the pot radiation half hourly diff, daily average, and daily average diff
    pot_rad_half_hourly, _, feature_name, _ = load_data("data/{}".format(file_name), ["PotRad"], [])
    half_hourly_diff, daily_avg, daily_diff = block_average_and_diff_expand(pot_rad_half_hourly, block_size=48)
    all_pot_rad_data = torch.cat((pot_rad_half_hourly, half_hourly_diff, daily_avg, daily_diff), 1)
    
    all_feature_names.extend(feature_name)
    all_feature_names.extend(["PotRadHalfHourlyDiff", "PotRadDailyAvg", "PotRadDailyDiff"])


    # 4. For wind direction, convert degrees (0 to 360) into sin/cos representation
    wd, _, feature_name, _ = load_data("data/{}".format(file_name), ["WD"], []);
    wd_cos_sin = wind_direction_to_cos_sin(wd) # returns an N x 2 torch tensor. Column [:,0] is cos, [:,1] is sin representation.
    all_feature_names.extend(["WD_COS", "WD_SIN"])

    
    # 5. Calculate GPP_prox, and nightly NEE average using SW_IN and the NEE.
    #    The values we receive are NOT NORMALIZED, and later-in-the-pipeline will be normalized.  
    sw_in, _, feature_name, _ = load_data("data/{}".format(file_name), ["SW_IN"], [])
    nee, _, feature_name, _ = load_data("data/{}".format(file_name), ["NEE"], [])
    gpp_prox_and_nightly_nee_average = compute_gpp_prox_and_nightly_nee_avg(sw_in, nee)
    all_feature_names.extend(["GPP_PROX", "NIGHTLY_NEE_AVG"])

    return torch.cat((doy_cos_sin, measured_features_tensor_raw, all_pot_rad_data, wd_cos_sin, gpp_prox_and_nightly_nee_average), 1), \
            all_feature_names
    

def load_and_clean_csv(file_path: str, drop_value: float = -9999.0) -> pd.DataFrame:
    """
    Loads a CSV file, drops rows where any value equals `drop_value`.

    Args:
        file_path (str): Path to the CSV file
        drop_value (float): Value to treat as missing (default: -9999)

    Returns:
        pd.DataFrame: Cleaned DataFrame with no -9999s
    """
    df = pd.read_csv(file_path)

    # Drop rows where any column contains the drop_value
    clean_df = df[~(df == drop_value).any(axis=1)].reset_index(drop=True)

    return clean_df

def split_data(gpp_inputs, reco_inputs, true_nee, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    assert gpp_inputs.shape[0] == reco_inputs.shape[0] == true_nee.shape[0], "Inputs must have same number of rows"
    
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
            'nee': true_nee[train_idx]
        },
        'val': {
            'gpp': gpp_inputs[val_idx],
            'reco': reco_inputs[val_idx],
            'nee': true_nee[val_idx]
        },
        'test': {
            'gpp': gpp_inputs[test_idx],
            'reco': reco_inputs[test_idx],
            'nee': true_nee[test_idx]
        }
    }


pre_processing = False
drop_na = False
normalize_raw_features = False
file_name = "CADSM_nee_partition_202101010000_202512312359.csv"
processed_file_name = "Processed_{}".format(file_name)
clean_file_name = "Cleaned_{}".format(file_name) # Will hold the rows that doesn't have NaN values

# ONLY NORMALIZE THE CLEAN FILE. Otherwise -9999's will affect the normalization.
normalized_file_name = "Normalized_{}".format(file_name)

if pre_processing:
    print("pre-processing the third stage file to obtain/calculate the necessary features")
    prepped_data, feature_names = prepare_data_using_csv(file_name)
    print("prepped data shape was", prepped_data.shape)
    print("feature_names", feature_names)


    X_df = pd.DataFrame(prepped_data.numpy(), columns=feature_names)
    processed_file_name = "Processed_{}".format(file_name)
    X_df.to_csv(processed_file_name, index=False)


if drop_na:
    # load_and_clean will drop all rows at least one missing value (ie. -9999)
    df_clean = load_and_clean_csv(processed_file_name)

    print(f"Original rows: {len(pd.read_csv(processed_file_name))}")
    print(f"Cleaned rows:  {len(df_clean)}")
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
    # all feature_names ['DOY_sin', 'DOY_cos', 'NEE', 'SW_IN', 'VPD', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'WD_COS', 'WD_SIN', 'GPP_PROX', 'NIGHTLY_NEE_AVG']
    
    raw_feature_names =  ['NEE', 'SW_IN', 'VPD', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'GPP_PROX', 'NIGHTLY_NEE_AVG']
    # WHEN NORMALIZING RAW VALUES, IF YOUR DATA SET ALREADY HAS THE DOY_SIN AND DOY_COS, YOU WANT TO SET prep_doy_sin_cos TO FALSE.
    raw_features, _, raw_feature_name, _ = load_data("{}".format(clean_file_name), raw_feature_names, [], prep_doy_sin_cos = False);
    
    # Normalize Raw Features:
    normalized_raw_features, _, _ = normalize_features(raw_features)

    already_normalized_feature_names = ['DOY_sin', 'DOY_cos', 'WD_COS', 'WD_SIN']
    already_normalized_features, _, already_normalized_feature_names, _ = \
            load_data("{}".format(clean_file_name), already_normalized_feature_names, [], prep_doy_sin_cos = False);

    
    all_feature_names = []
    all_feature_names.extend(already_normalized_feature_names)
    all_feature_names.extend(raw_feature_names)

    all_features = torch.cat((already_normalized_features, normalized_raw_features), 1)

    df_normalized = pd.DataFrame(all_features.numpy(), columns=all_feature_names)
    df_normalized.to_csv(normalized_file_name, index=False)    




# Read normalized values file then do backprop magic time.
# all feature_names ['DOY_sin', 'DOY_cos', 'NEE', 'SW_IN', 'VPD', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'WD_COS', 'WD_SIN', 'GPP_PROX', 'NIGHTLY_NEE_AVG']
GPP_INPUT_FEATURES = ['SW_IN', 'VPD', 'TA', 'WTD', 'WS', 'PotRad', 'PotRadHalfHourlyDiff', 'PotRadDailyAvg', 'PotRadDailyDiff', 'WD_COS', 'WD_SIN', 'GPP_PROX']
RECO_INPUT_FEATURES = ['DOY_sin', 'DOY_cos', 'TA', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'WTD', 'WS', 'WD_COS', 'WD_SIN', 'NIGHTLY_NEE_AVG']
NEE = ['NEE']

# normalized_file_name
gpp_inputs, _, gpp_input_names, _ = load_data("{}".format(normalized_file_name), GPP_INPUT_FEATURES, [], prep_doy_sin_cos = False)
reco_inputs, _, reco_input_names, _ = load_data("{}".format(normalized_file_name), RECO_INPUT_FEATURES, [], prep_doy_sin_cos = False)
true_nee, _, true_nee_name, _ = load_data("{}".format(normalized_file_name), NEE, [], prep_doy_sin_cos = False)
print(f"gpp_input_names,  {gpp_input_names} \n"
      f"reco_input_names,  {reco_input_names} \n"
      f"true_nee_name,  {true_nee_name} \n")

# fit(gpp_inputs=gpp_inputs, reco_inputs=reco_inputs, true_nee=true_nee)

# Not sure if this is the cleanest way. But keep for now as we need to validate.
splits = split_data(gpp_inputs, reco_inputs, true_nee, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

# X_gpp_train = splits['train']['gpp']
# X_reco_train = splits['train']['reco']
# y_train = splits['train']['nee']

# X_gpp_val = splits['val']['gpp']
# X_reco_val = splits['val']['reco']
# y_val = splits['val']['nee']

# X_gpp_test = splits['test']['gpp']
# X_reco_test = splits['test']['reco']
# y_test = splits['test']['nee']
gpp_model, reco_model = better_fit(
    X_gpp_train=splits['train']['gpp'],
    X_reco_train=splits['train']['reco'],
    y_train=splits['train']['nee'],
    X_gpp_val=splits['val']['gpp'],
    X_reco_val=splits['val']['reco'],
    y_val=splits['val']['nee'],
)




"""

All data just training used.
Epoch 9999, Loss: 0.0014
R²: 0.9495583772659302

gpp_inputs = torch.randn(32, input_dim_gpp)
reco_inputs = torch.randn(32, input_dim_reco)
true_nee = torch.randn(32, 1)  # Measured NEE

    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200)
"""