import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import pandas as pd
from datetime import datetime



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
    prep_doy_sin_cos: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the dataset, processes it, and returns input and target tensors.

    Args:
        file_path (str): Path to the CSV file.
        input_features (List[str]): Features to be used as input to the model.
        target_features (List[str]): Features to be used as targets for loss calculation.
        dropna (bool): If True, drop rows with NaN in selected columns.
        prep_doy_sin_cos (bool): If True, prep the DOY sin cos vals based on DATE (Dates are formatted as YYYY-MM-DD)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
    """
    # Load the CSV
    df = pd.read_csv(file_path)

    # ignore for now
    # Replace -9999 with NaN
    # df.replace(-9999, pd.NA, inplace=True)
    
    if prep_doy_sin_cos:
        df["DOY_sin"], df["DOY_cos"] = compute_doy_sin_cos(df["DATE"])
        torch.set_printoptions(profile="full")
        torch.set_printoptions(linewidth=200)

        # print(df["DOY_sin"], df["DOY_cos"])


    # Select only the required columns
    data = df[input_features + target_features]

    # Optionally drop rows with missing values
    if dropna:
        data = data.dropna()

    # Split into input and target
    inputs = data[input_features].astype(float).values
    targets = data[target_features].astype(float).values

    # Convert to PyTorch tensors
    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)

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
    # plt.scatter(range(len(measurement2)), measurement2, label='Measurement 2', color='orange', s=10, alpha=0.8)

    
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
    plt.plot(measurement1, label='Measurement 1', color='blue', linewidth=1, alpha=0.8)
    plt.plot(measurement2, label='Measurement 2', color='orange', linewidth=1, alpha=0.8)
    plt.plot(measurement3, label='Measurement 3', color='green', linewidth=1, alpha=0.8)
    plt.plot(measurement4, label='Measurement 4', color='red', linewidth=1, alpha=0.8)

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
        block_size (int): Block size (e.g., 48 for half-hourly data to get daily stats)

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




# file_name = "AMF_CA-DSM_BASE_HH_local_prepping.csv"
file_name = "CADSM_nee_partition_202101010000_202512312359.csv"

# GPP NN INPUT AND TARGET FEATURES.
gpp_input_features = ["NEE_PI_SC_JSZ_MAD_RP_uStar_orig", "NEE_PI_SC_JSZ_MAD_RP_uStar_f"]
gpp_target_features = ["NEE", "GPP_U95_f"]
gpp_input_tensor, gpp_target_tensor = load_data("data/{}".format(file_name), gpp_input_features, gpp_target_features)
print("jpp", gpp_input_tensor.shape, gpp_target_tensor.shape)


# RECO NN INPUT AND TARGET FEATURES.
reco_input_features = ["DOY_sin", "DOY_cos"]
reco_target_features = ["NEE"]
reco_input_tensor, reco_target_tensor = load_data("data/{}".format(file_name), reco_input_features, reco_target_features)
print("reco", reco_input_tensor.shape, reco_target_tensor.shape)



torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=200)

# getting  the block_average_and_diff_expand
# if block size 48, then makes sense to name it half_hourly_diff, or it could be made hourly for block size 24.
# half_hourly_diff, daily_avg, daily_diff = block_average_and_diff_expand(reco_input_tensor, block_size=48)
# all_catted = torch.cat((reco_input_tensor, half_hourly_diff, daily_avg, daily_diff), 1)
# print("okocer final vers Jun 16")
# print(all_catted.size())
# quad_plotter(all_catted)


print(reco_input_tensor[0::48])
# import pdb; pdb.set_trace()
# DOY_sin, DOY_cos = get_doy_sin_cos(reco_input_tensor)

# print(f" The PotRad_U95 and PotRad_uStar tensor was: {reco_input_tensor.t()}")
# print(gpp_input_tensor)
# pair_plotter(gpp_input_tensor)