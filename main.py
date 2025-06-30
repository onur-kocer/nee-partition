import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Union
import pandas as pd
from datetime import datetime
import math


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


def fit(): 
  # Instantiate models
  input_dim_gpp = 10  # adjust based on your actual input features
  input_dim_reco = 5

  gpp_model = SNN_GPP(input_dim_gpp)
  reco_model = SNN_RECO(input_dim_reco)

  # Optimizers (can be separate or joint)
  optimizer = optim.Adam(list(gpp_model.parameters()) + list(reco_model.parameters()), lr=1e-3)

  # Loss function
  criterion = nn.MSELoss()

  # Example training loop
  for epoch in range(100):
      gpp_model.train()
      reco_model.train()

      # Batch of inputs (replace with your actual data loader)
      gpp_inputs = torch.randn(32, input_dim_gpp)
      reco_inputs = torch.randn(32, input_dim_reco)
      true_nee = torch.randn(32, 1)  # Measured NEE

      # Forward pass
      gpp_pred = gpp_model(gpp_inputs)
      reco_pred = reco_model(reco_inputs)
      nee_pred = gpp_pred + reco_pred

      # Loss and backward
      loss = criterion(nee_pred, true_nee)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


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

    # Optionally drop rows with missing values. Not tested yet.
    if dropna:
        data = data.dropna()

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
    # ALL NECESSARY GPP FEATURES ["SW_IN", "PotRad", "VPD", "TA", "TS_1", "TS_2", "TS_3", "TS_4", "WTD", "WS", "WD", "NEE"]
    # ALL NECESSARY RECO FEATURES = ["TA", "TS_1", "TS_2", "TS_3", "TS_4", "WTD", "WS", "WD", "NEE"]
    
    # 1. Collect all variables that haven't been normalized yet. 
    #    These will be normalized later once the gaps have been dealt with.
    measured_features_raw = ["NEE", "SW_IN", "VPD", "TA", "TS_1", "TS_2", "TS_3", "TS_4", "WTD", "WS"] 
    measured_features_tensor_raw, _, feature_name, _ = load_data("data/{}".format(file_name), measured_features_raw, [])
    all_feature_names.extend(feature_name)


    # 2. Prep the pot radiation half hourly diff, daily average, and daily average diff
    pot_rad_half_hourly, _, feature_name, _ = load_data("data/{}".format(file_name), ["PotRad"], [])
    half_hourly_diff, daily_avg, daily_diff = block_average_and_diff_expand(pot_rad_half_hourly, block_size=48)
    all_pot_rad_data = torch.cat((pot_rad_half_hourly, half_hourly_diff, daily_avg, daily_diff), 1)
    
    all_feature_names.extend(feature_name)
    all_feature_names.extend(["PotRadHalfHourlyDiff", "PotRadDailyAvg", "PotRadDailyDiff"])


    # 3. For wind direction, convert degrees (0 to 360) into sin/cos representation
    wd, _, feature_name, _ = load_data("data/{}".format(file_name), ["WD"], []);
    wd_cos_sin = wind_direction_to_cos_sin(wd) # returns an N x 2 torch tensor. Column [:,0] is cos, [:,1] is sin representation.
    all_feature_names.extend(["WD_COS", "WD_SIN"])

    
    # 4. Calculate GPP_prox, and nightly NEE average using SW_IN and the NEE
    sw_in, _, feature_name, _ = load_data("data/{}".format(file_name), ["SW_IN"], [])
    nee, _, feature_name, _ = load_data("data/{}".format(file_name), ["NEE"], [])
    gpp_prox_and_nightly_nee_average = compute_gpp_prox_and_nightly_nee_avg(sw_in, nee)
    all_feature_names.extend(["GPP_PROX", "NIGHTLY_NEE_AVG"])


    # 5. Calculate Day of Year Cos/Sine
    doy_cos_sin, _, feature_name, _ = load_data("data/{}".format(file_name), ["DOY_sin", "DOY_cos"] , [])
    all_feature_names.extend(feature_name)

    print("GPP CSV tensors", measured_features_tensor_raw.shape)

    # import pdb; pdb.set_trace()
    return torch.cat((all_pot_rad_data, wd_cos_sin, gpp_prox_and_nightly_nee_average, doy_cos_sin, measured_features_tensor_raw), 1), \
            all_feature_names
    

    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=200)
    # NORMALIZATION
    # gpp_input_tensor_norm, _, _ = normalize_features(gpp_input_tensor_raw)
    # # print(reco_input_tensor[0::48]) # printing every 48th element

file_name = "CADSM_nee_partition_202101010000_202512312359.csv"
prepped_data, feature_names = prepare_data_using_csv(file_name)
print("prepped data shape was", prepped_data.shape)
print("feature_names", feature_names)

X_df = pd.DataFrame(prepped_data.numpy(), columns=feature_names)
X_df.to_csv("processed_data.csv", index=False)





# X = torch.tensor([[1.0, 100.0],
#                   [2.0, 300.0],
#                   [3.0, 450.0]])

# X_norm, X_min, X_max = normalize_features(X)

# print("X_norm:\n", X_norm)
# print("X_min:", X_min)
# print("X_max:", X_max)