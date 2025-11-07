import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def calculate_acf (clean_file_name):
    # ===============================================================
    # LOAD AND PREPARE DATA
    # ===============================================================
    # use the clean_file_name as that one has the gaps.
    df = pd.read_csv(clean_file_name)   # columns: Year, Month, Day, TIME, NEE
    # other_features, other_feature_names = load_data("{}".format(processed_file_name), other_feature_names)
    print(df)
    df = df.sort_values(by=["Year", "Month", "Day", "TIME"]).reset_index(drop=True)

    # Build datetime index
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]]) + \
                    pd.to_timedelta(df["TIME"] * 60, unit="m")
    df = df.set_index("datetime")

    # Ensure half-hourly grid
    expected_freq = "30T"
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
    s = df["NEE"].reindex(full_index)

    # Identify missing data
    is_missing = s.isna()

    # ===============================================================
    # SPLIT INTO CONTIGUOUS SEGMENTS
    # ===============================================================
    segments = []
    current_segment = []

    for val, missing in zip(s, is_missing):
        if not missing:
            current_segment.append(val)
        else:
            if current_segment:
                segments.append(np.array(current_segment))
                current_segment = []
    if current_segment:
        segments.append(np.array(current_segment))

    # Optionally filter out very short fragments (<3 days)
    # segments = [seg for seg in segments if len(seg) >= 3 * 48]

    print(f"Found {len(segments)} contiguous segments.")
    print(f"Mean segment length: {np.mean([len(x) for x in segments]):.0f} points")
    print(f"Median segment length: {np.median([len(x) for x in segments]):.0f} points")
    print(f"Mode segment length: {pd.Series([len(x) for x in segments]).mode().iloc[0]} points")
    # ===============================================================
    # Seg length trends
    # ===============================================================

    # Compute segment lengths
    segment_lengths = np.array([len(x) for x in segments])

    # Descriptive statistics
    desc = pd.Series(segment_lengths).describe()
    print("\nSegment length descriptives:")
    print(desc)

    # Frequency table (top 10)
    freq_table = pd.Series(segment_lengths).value_counts().sort_index()
    print("\nFrequency table (first 10 entries):")
    print(freq_table.head(10))

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(segment_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Contiguous Segment Lengths")
    plt.xlabel("Segment length (number of points)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # Optional: Cumulative distribution plot (CDF)
    plt.figure(figsize=(10, 5))
    sorted_lengths = np.sort(segment_lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    plt.plot(sorted_lengths, cdf, lw=2)
    plt.title("Cumulative Distribution of Segment Lengths")
    plt.xlabel("Segment length (number of points)")
    plt.ylabel("Cumulative proportion")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


    # ===============================================================
    # DEFINE MULTIPLE LAG WINDOWS
    # ===============================================================
    points_per_day = 48  # half-hours/day
    lag_windows = {
        "1 day": 1 * points_per_day,
        "3 days": 3 * points_per_day,
        # "1 week": 7 * points_per_day,
        # "10 days": 10 * points_per_day,
        # "14 days": 14 * points_per_day,
        # "1 month (~30d)": 30 * points_per_day,
        # "1.5 months (~45)": 45 * points_per_day,
        # "3 months (~90d)": 90 * points_per_day,
        # "1 year (~365d)": 365 * points_per_day,
    }

    # ===============================================================
    # COMPUTE AND PLOT ACFs
    # ===============================================================
    plt.figure(figsize=(12, 6))

    for label, max_lag in lag_windows.items():
        acfs = []
        for seg in segments:
            if len(seg) > max_lag:
                acf_vals = acf(seg, nlags=max_lag, fft=True)
                acfs.append(acf_vals)
        if len(acfs) == 0:
            print(f"⚠️ No segments long enough for {label} (need > {max_lag} points)")
            continue

        mean_acf = np.mean(acfs, axis=0)
        lags = np.arange(len(mean_acf))

        plt.plot(lags / points_per_day, mean_acf, label=label)

    plt.axhline(0, color='black', lw=0.7)
    plt.xlabel("Lag (days)")
    plt.ylabel("Mean ACF")
    plt.title("ACF of NEE for Multiple Lag Windows")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===============================================================
    # OPTIONAL: Report decorrelation lag for each window
    # ===============================================================
    threshold = 0.1
    for label, max_lag in lag_windows.items():
        acfs = []
        for seg in segments:
            if len(seg) > max_lag:
                acf_vals = acf(seg, nlags=max_lag, fft=True)
                acfs.append(acf_vals)
        if len(acfs) == 0:
            continue
        mean_acf = np.mean(acfs, axis=0)
        below = np.where(np.abs(mean_acf) < threshold)[0]
        if len(below) > 0:
            decor = below[0]
            print(f"{label}: decorrelation lag ≈ {decor} half-hours ({decor/points_per_day:.2f} days)")
        else:
            print(f"{label}: ACF never dropped below {threshold} within {max_lag/points_per_day:.1f} days.")
    
def split_data(
    gpp_inputs, reco_inputs, true_nee, time, sw_in_raw,
    Year, Month, Day,
    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    seed=42, split_by='point'  # options: 'point', 'day', 'week'
):
    """
    Splits data into train/val/test sets either by data points, days, or weeks.

    Args:
        gpp_inputs, reco_inputs, true_nee, time, sw_in_raw: torch tensors (N,)
        Year, Month, Day: arrays or tensors with same length as inputs
        split_by: one of {'point', 'day', 'week'}
        train_ratio, val_ratio, test_ratio: float ratios that sum to 1
        seed: random seed for reproducibility
    """

    # --- Consistency check ---
    N = gpp_inputs.shape[0]
    assert all(x.shape[0] == N for x in [reco_inputs, true_nee, time, sw_in_raw, Year, Month, Day]), \
        "All inputs must have the same number of rows."

    torch.manual_seed(seed)

    # Convert torch tensors or lists to numpy arrays
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().flatten()
        return x
    # --- Create group IDs based on desired split type ---
    if split_by == 'day':
        df = pd.DataFrame({
            'Year': to_np(Year),
            'Month': to_np(Month),
            'Day': to_np(Day)
        })
        groups = df.groupby(['Year', 'Month', 'Day']).ngroup()
    elif split_by == 'week':
        # Combine into a date and use ISO week numbers
        df = pd.DataFrame({
            'Year': to_np(Year),
            'Month': to_np(Month),
            'Day': to_np(Day)
        })        
        dates = pd.to_datetime(df[['Year', 'Month', 'Day']])
        week_ids = dates.dt.isocalendar().week
        groups = (df['Year'].astype(str) + '_' + week_ids.astype(str)).astype('category').cat.codes
    else:
        # Default: each point is its own group
        # TODO not sure if this is working.
        groups = torch.arange(N)

    # --- Unique groups for splitting ---
    unique_groups = torch.tensor(pd.unique(groups))
    num_groups = len(unique_groups)

    # Shuffle groups
    perm = torch.randperm(num_groups)
    unique_groups = unique_groups[perm]

    # Compute split sizes
    n_train = int(num_groups * train_ratio)
    n_val = int(num_groups * val_ratio)
    n_test = num_groups - n_train - n_val

    # Assign groups
    train_groups = unique_groups[:n_train]
    val_groups = unique_groups[n_train:n_train + n_val]
    test_groups = unique_groups[n_train + n_val:]

    # Map group IDs to indices
    group_tensor = torch.tensor(groups)
    train_idx = torch.isin(group_tensor, train_groups).nonzero(as_tuple=True)[0]
    val_idx = torch.isin(group_tensor, val_groups).nonzero(as_tuple=True)[0]
    test_idx = torch.isin(group_tensor, test_groups).nonzero(as_tuple=True)[0]

    # --- Return split tensors ---
    def subset(idx):
        return {
            'gpp': gpp_inputs[idx],
            'reco': reco_inputs[idx],
            'nee': true_nee[idx],
            'sw_in_raw': sw_in_raw[idx],
            'time': time[idx],
            'year': Year[idx],
            'month': Month[idx],
            'day': Day[idx]
        }

    return {
        'train': subset(train_idx),
        'val': subset(val_idx),
        'test': subset(test_idx)
    }

def filter_by_threshold(sw_in_raw, other_tensor, threshold=10):
    """
    Keep only rows where sw_in_raw <= threshold.
    This will get return the night data.

    sw_in_raw: 1D torch tensor of floats/ints
    other_tensor: torch tensor (same first dimension as sw_in_raw)
    threshold: numeric value

    Returns:
        sw_in_raw_filtered, other_tensor_filtered
    """
    mask = sw_in_raw <= threshold
    return sw_in_raw[mask], other_tensor[mask]


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

    x_abs_max = torch.maximum(abs(X_min), abs(X_max)).clamp(min=1e-8)
    X = X_norm * x_abs_max

    return X