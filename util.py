import pandas as pd
import numpy as np
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
    
