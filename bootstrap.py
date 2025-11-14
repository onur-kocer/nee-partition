import torch
import numpy as np
import pandas as pd
import os
# from datetime import datetime
import util
from util import split_data, filter_by_threshold, unnormalize_features



def bootstrap_confidence_interval(data, confidence=0.95):
    """Compute mean and percentile-based confidence interval."""
    arr = np.array(data)
    mean = np.mean(arr)
    lower = np.percentile(arr, ((1 - confidence) / 2) * 100)
    upper = np.percentile(arr, (1 - (1 - confidence) / 2) * 100)
    return mean, (lower, upper)

def bootstrap_evaluation(
    gpp_inputs,
    reco_inputs,
    true_nee,
    time,
    sw_in_raw,
    year,
    month,
    day,
    tramontana_run,
    hidden_size,
    GPP_INPUT_FEATURES,
    RECO_INPUT_FEATURES,
    site_name="SITE",
    run_type_str="DEMO",
    num_bootstraps=200,
    save_dir="./bootstrap_results",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Perform bootstrapping on training data and evaluate on a fixed test set.

    Returns:
        dict with mean & 95% CI for RMSE_night, GPP, RECO, NEE
    """
    from main import better_fit_gpu, evaluate_single_model
    # 1. Split once
    splits = split_data(
        gpp_inputs, reco_inputs, true_nee, time, sw_in_raw,
        year, month, day,
        train_ratio=0.8, val_ratio=0.0, test_ratio=0.2,
        split_by="point"
    )

    train_set = splits["train"]
    test_set = splits["test"]

    n_train = len(train_set["nee"])
    test_sw_in = test_set["sw_in_raw"]

    # Prepare save dir
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f"{site_name}_bootstrap_results.csv")

    # Prepare storage
    all_results = []

    print(f"Bootstrapping {num_bootstraps} models on {device.upper()} ...")


    for b in range(num_bootstraps):
        # Use the bootstrap iteration number as the random seed for reproducibility of bootstrap resamples
        torch.manual_seed(b + 42)
        print(f"→ Bootstrap {b+1}/{num_bootstraps}")

        # --- Resample train indices ---
        resample_idx = torch.randint(0, n_train, (n_train,))

        # --- Identify OOB (out-of-bag) indices ---
        all_idx = torch.arange(n_train)
        oob_mask = torch.ones(n_train, dtype=torch.bool)
        oob_mask[resample_idx] = False
        oob_idx = all_idx[oob_mask]  # indices not selected for this bootstrap        

        # --- Build bootstrap training set ---
        X_gpp_train = train_set["gpp"][resample_idx].to(device)
        X_reco_train = train_set["reco"][resample_idx].to(device)
        y_train = train_set["nee"][resample_idx].to(device)
        SW_IN_RAW_train = train_set["sw_in_raw"][resample_idx].to(device)

        # OOB validation set
        X_gpp_val = train_set["gpp"][oob_idx].to(device)
        X_reco_val = train_set["reco"][oob_idx].to(device)
        y_val = train_set["nee"][oob_idx].to(device)
        SW_IN_RAW_val = train_set["sw_in_raw"][oob_idx].to(device)        

        # --- Train model on resampled data ---
        gpp_model, reco_model, _ = better_fit_gpu(
            tram=tramontana_run,
            hidden_layer_size=hidden_size,
            X_gpp_train=X_gpp_train,
            X_reco_train=X_reco_train,
            y_train=y_train,
            X_gpp_val=X_gpp_val,
            X_reco_val=X_reco_val,
            y_val=y_val,
            SW_IN_RAW_train=SW_IN_RAW_train,
            SW_IN_RAW_val=SW_IN_RAW_val,
        )

        # --- Save the trained bootstrap model. ---
        # Remark: The saving part can be omitted in the future. It is not absolutely necessary.
        torch.save(gpp_model.state_dict(), f"trained_models/{site_name}_gpp_model_{run_type_str}.pth")
        torch.save(reco_model.state_dict(), f"trained_models/{site_name}_reco_model_{run_type_str}.pth")        

        # --- Evaluate on fixed test set ---
        gpp_pred, reco_pred, nee_pred, nee_min, nee_max = evaluate_single_model(
            test_set["gpp"].to(device),
            test_set["reco"].to(device),
            test_set["time"],
            test_set["sw_in_raw"].to(device),
            model_inputs_information=f"Bootstrap {b+1}",
            save_plot=False,
            plot_saving_str=None,
            results_dict={},
            experiment_id=b+1,
            GPP_INPUT_FEATURES=GPP_INPUT_FEATURES,
            RECO_INPUT_FEATURES=RECO_INPUT_FEATURES,
            splits=splits,
            is_bootstrap_eval = True
        )

        # # --- Compute derived quantities ---
        # nee_pred = reco_pred - gpp_pred

        # --- Filter nighttime data ---
        # TODO BUG HERE. test_set["nee"] is the normalized NEE!!! that must be unnormalized.
        # import pdb; pdb.set_trace()
        # Have to unnormalize the test_set["nee"]
        raw_test_set_nee = unnormalize_features(test_set["nee"], nee_min, nee_max)

        # _, true_nee_night = filter_by_threshold(test_sw_in, test_set["nee"], threshold=10)
        _, true_nee_night = filter_by_threshold(test_sw_in, raw_test_set_nee, threshold=10)
        _, pred_nee_night = filter_by_threshold(test_sw_in, nee_pred, threshold=10)

        # --- RMSE_night ---
        rmse_night = torch.sqrt(torch.mean((pred_nee_night - true_nee_night) ** 2)).item()

        # --- Store results for CI ---
        all_results.append({
            "bootstrap_id": b + 1,
            "rmse_night": rmse_night,
            "gpp_mean": torch.mean(gpp_pred).item(),
            "reco_mean": torch.mean(reco_pred).item(),
            "nee_mean": torch.mean(nee_pred).item(),
        })

        # Save interim every 20 iterations
        if (b + 1) % 20 == 0:
            pd.DataFrame(all_results).to_csv(results_path, index=False)

    # Final save
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(results_path, index=False)

    # --- Compute statistics ---
    mean_rmse, ci_rmse = bootstrap_confidence_interval(df_results["rmse_night"])
    mean_gpp, ci_gpp = bootstrap_confidence_interval(df_results["gpp_mean"])
    mean_reco, ci_reco = bootstrap_confidence_interval(df_results["reco_mean"])
    mean_nee, ci_nee = bootstrap_confidence_interval(df_results["nee_mean"])

    print("\n✅ Bootstrapping complete.")
    print(f"Results saved to {results_path}")

    summary = {
        "RMSE_night": (mean_rmse, ci_rmse),
        "GPP": (mean_gpp, ci_gpp),
        "RECO": (mean_reco, ci_reco),
        "NEE": (mean_nee, ci_nee)
    }

    print(f"\nBootstrapped summary ({num_bootstraps} samples):")
    print(f"RMSE_night: Mean = {mean_rmse:.3f}, 95% CI = [{ci_rmse[0]:.3f}, {ci_rmse[1]:.3f}]")
    print(f"GPP Mean   = {mean_gpp:.3f}, 95% CI = [{ci_gpp[0]:.3f}, {ci_gpp[1]:.3f}]")
    print(f"RECO Mean  = {mean_reco:.3f}, 95% CI = [{ci_reco[0]:.3f}, {ci_reco[1]:.3f}]")
    print(f"NEE Mean   = {mean_nee:.3f}, 95% CI = [{ci_nee[0]:.3f}, {ci_nee[1]:.3f}]")

    return summary
