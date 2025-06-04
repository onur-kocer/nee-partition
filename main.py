import torch
import torch.nn as nn
import torch.optim as optim

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



import pandas as pd
import torch
from typing import List, Tuple

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


file_name = "AMF_CA-DSM_BASE_HH_local_prepping.csv"

gpp_input_features = ["TA_1_1_1", "RH_1_1_1", "VPD_1_1_1", "COND_WATER_1_1_1"]
gpp_target_features = ["NEE", "GPP_U95_f"]

gpp_input_tensor, gpp_target_tensor = load_data("data/{}".format(file_name), gpp_input_features, gpp_target_features)
print(gpp_input_tensor.shape, gpp_target_tensor.shape)


reco_input_features = ["SW_IN_1_1_1"]
reco_target_features = ["NEE"]

reco_input_tensor, reco_target_tensor = load_data("data/{}".format(file_name), reco_input_features, reco_target_features)
print(reco_input_tensor.shape, reco_target_tensor.shape)

torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=200)
print(f" The SW_IN_1_1_1 tensor was: {reco_input_tensor.t()}")

