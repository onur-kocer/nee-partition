import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

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





def load_data(
    file_path: str,
    input_features: List[str],
    target_features: List[str],
    dropna: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the dataset, processes it, and returns input and target tensors.

    Args:
        file_path (str): Path to the CSV file.
        input_features (List[str]): Features to be used as input to the model.
        target_features (List[str]): Features to be used as targets for loss calculation.
        dropna (bool): If True, drop rows with NaN in selected columns.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
    """
    # Load the CSV
    df = pd.read_csv(file_path)
    # df = pd.read_csv(file_path, delimiter=",", engine="python")

    # ignore for now
    # Replace -9999 with NaN
    # df.replace(-9999, pd.NA, inplace=True)

    # Ignore for now.
    # # Convert timestamps to datetime
    # df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format="%Y%m%d%H%M")
    # df["TIMESTAMP_END"] = pd.to_datetime(df["TIMESTAMP_END"], format="%Y%m%d%H%M")

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
    # measurement1 = data[:, 0].numpy()
    measurement2 = data[:, 1].numpy()

    plt.figure(figsize=(15, 5))
    # to see as individual lines
    # plt.plot(measurement1, label='Measurement 1', color='blue', linewidth=1,  alpha=0.8)
    plt.plot(measurement2, label='Measurement 2', color='orange', linewidth=1,  alpha=0.8)
    
    # for comparing data distribution
    # plt.hist(measurement1, bins=100, alpha=0.5, label='Measurement 1')
    # plt.hist(measurement2, bins=100, alpha=0.5, label='Measurement 2')

    plt.legend()
    plt.title("Time-Series of Measurements")
    plt.xlabel("Time")
    plt.ylabel("Measurement Value")
    plt.grid(True)
    plt.show()



def block_average_and_diff_expand(data: torch.Tensor, block_size: int):
    """
    Computes blockwise averages and differences between consecutive blocks,
    then expands both to the original shape.

    Args:
        data (torch.Tensor): Input tensor of shape [N, D]
        block_size (int): Size of each block (e.g., 48 for daily averages if half-hourly)

    Returns:
        avg_expanded (torch.Tensor): Tensor of shape [N, D] with blockwise averages repeated
        diff_expanded (torch.Tensor): Tensor of shape [N, D] with blockwise day-to-day differences repeated
    """
    N, D = data.shape
    assert N % block_size == 0, "Data length must be divisible by block size"
    num_blocks = N // block_size

    # Step 1: Reshape to [num_blocks, block_size, D]
    reshaped = data.view(num_blocks, block_size, D)

    # Step 2: Compute block averages [num_blocks, D]
    block_means = reshaped.mean(dim=1)  # shape: [num_blocks, D]

    # Step 3: Compute daily differences [num_blocks, D]
    # First difference will be 0 or can be NaN if preferred
    block_diffs = torch.zeros_like(block_means)
    block_diffs[1:] = block_means[1:] - block_means[:-1]

    # Step 4: Expand both [num_blocks, D] -> [num_blocks, block_size, D]
    avg_expanded = block_means.unsqueeze(1).expand(-1, block_size, -1)
    diff_expanded = block_diffs.unsqueeze(1).expand(-1, block_size, -1)

    # Step 5: Reshape back to [N, D]
    return avg_expanded.reshape(N, D), diff_expanded.reshape(N, D)


# file_name = "AMF_CA-DSM_BASE_HH_local_prepping.csv"
file_name = "CADSM_nee_partition_202101010000_202512312359.csv"

# gpp_input_features = ["TA_1_1_1", "RH_1_1_1", "VPD_1_1_1", "COND_WATER_1_1_1"]
# gpp_target_features = ["NEE", "GPP_U95_f"]

# gpp_input_tensor, gpp_target_tensor = load_data("data/{}".format(file_name), gpp_input_features, gpp_target_features)
# print(gpp_input_tensor.shape, gpp_target_tensor.shape)


reco_input_features = ["PotRad_uStar"]
reco_target_features = ["NEE"]

reco_input_tensor, reco_target_tensor = load_data("data/{}".format(file_name), reco_input_features, reco_target_features)
print(reco_input_tensor.shape, reco_target_tensor.shape)


torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=200)
# print(f" The PotRad_U95 and PotRad_uStar tensor was: {reco_input_tensor.t()}")


# pair_plotter(reco_input_tensor)
# measurement1 = data[:, 0].numpy()
# measurement2 = data[:, 1].numpy()
# torch.Size([86160, 1])


half_hourly_diff = torch.sub(reco_input_tensor[1:], reco_input_tensor[0:-1])
# pad the half_hourly_diffative by one extra entry at the beginning as the output is one entry smaller.
half_hourly_diff = torch.cat((torch.tensor([[0]]), half_hourly_diff), 0)
print(half_hourly_diff.size())

pair = torch.cat((reco_input_tensor, half_hourly_diff), 1)

# import pdb; pdb.set_trace()
# works.
# print(pair)
# pair_plotter(pair)


daily_avg, daily_diff = block_average_and_diff_expand(reco_input_tensor, block_size=48)


all_catted = torch.cat((reco_input_tensor, half_hourly_diff, daily_avg, daily_diff), 1)
print(all_catted.size())
print(all_catted)
# print(torch.cat((reco_input_tensor, half_hourly_diff, daily_avg, daily_diff)), 1)