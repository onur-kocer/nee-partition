def compare_two_models ():
  print(f"Hello!")

dict = {
    "experiment_0": {
        "run_type": "Custom",
        "gpp_inputs": [
            "VPD",
            "TA",
            "WTD",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.38,
                "rmse": 5.62
            },
            "NT_GPP_vs_model": {
                "r2": 0.35,
                "rmse": 5.7
            },
            "DT_RECO_vs_model": {
                "r2": -4.67,
                "rmse": 5.53
            },
            "NT_RECO_vs_model": {
                "r2": -6.22,
                "rmse": 5.42
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_1": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "TA",
            "WTD",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.8,
                "rmse": 3.17
            },
            "NT_GPP_vs_model": {
                "r2": 0.78,
                "rmse": 3.35
            },
            "DT_RECO_vs_model": {
                "r2": -0.64,
                "rmse": 2.97
            },
            "NT_RECO_vs_model": {
                "r2": -1.23,
                "rmse": 3.01
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_2": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "WTD",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.95,
                "rmse": 1.58
            },
            "NT_GPP_vs_model": {
                "r2": 0.94,
                "rmse": 1.8
            },
            "DT_RECO_vs_model": {
                "r2": 0.83,
                "rmse": 0.96
            },
            "NT_RECO_vs_model": {
                "r2": 0.83,
                "rmse": 0.84
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_3": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.89,
                "rmse": 2.33
            },
            "NT_GPP_vs_model": {
                "r2": 0.87,
                "rmse": 2.57
            },
            "DT_RECO_vs_model": {
                "r2": 0.26,
                "rmse": 2.0
            },
            "NT_RECO_vs_model": {
                "r2": -0.02,
                "rmse": 2.03
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_4": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WTD",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.96,
                "rmse": 1.52
            },
            "NT_GPP_vs_model": {
                "r2": 0.94,
                "rmse": 1.73
            },
            "DT_RECO_vs_model": {
                "r2": 0.82,
                "rmse": 0.97
            },
            "NT_RECO_vs_model": {
                "r2": 0.82,
                "rmse": 0.85
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_5": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WTD",
            "WS",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.18,
                "rmse": 6.48
            },
            "NT_GPP_vs_model": {
                "r2": 0.1,
                "rmse": 6.71
            },
            "DT_RECO_vs_model": {
                "r2": -6.95,
                "rmse": 6.55
            },
            "NT_RECO_vs_model": {
                "r2": -9.77,
                "rmse": 6.61
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_6": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WTD",
            "WS",
            "PotRad",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": -1.53,
                "rmse": 11.37
            },
            "NT_GPP_vs_model": {
                "r2": -1.68,
                "rmse": 11.58
            },
            "DT_RECO_vs_model": {
                "r2": -23.25,
                "rmse": 11.43
            },
            "NT_RECO_vs_model": {
                "r2": -31.75,
                "rmse": 11.54
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_7": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WTD",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.4,
                "rmse": 5.56
            },
            "NT_GPP_vs_model": {
                "r2": 0.34,
                "rmse": 5.74
            },
            "DT_RECO_vs_model": {
                "r2": -4.75,
                "rmse": 5.57
            },
            "NT_RECO_vs_model": {
                "r2": -6.71,
                "rmse": 5.6
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_8": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WTD",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "WD_COS",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.86,
                "rmse": 2.65
            },
            "NT_GPP_vs_model": {
                "r2": 0.82,
                "rmse": 2.97
            },
            "DT_RECO_vs_model": {
                "r2": -0.07,
                "rmse": 2.4
            },
            "NT_RECO_vs_model": {
                "r2": -0.6,
                "rmse": 2.55
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_9": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WTD",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_SIN",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.76,
                "rmse": 3.52
            },
            "NT_GPP_vs_model": {
                "r2": 0.72,
                "rmse": 3.74
            },
            "DT_RECO_vs_model": {
                "r2": -1.08,
                "rmse": 3.35
            },
            "NT_RECO_vs_model": {
                "r2": -1.9,
                "rmse": 3.43
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_10": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WTD",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "GPP_PROX"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.89,
                "rmse": 2.35
            },
            "NT_GPP_vs_model": {
                "r2": 0.87,
                "rmse": 2.6
            },
            "DT_RECO_vs_model": {
                "r2": 0.1,
                "rmse": 2.2
            },
            "NT_RECO_vs_model": {
                "r2": -0.18,
                "rmse": 2.19
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    },
    "experiment_11": {
        "run_type": "Custom",
        "gpp_inputs": [
            "SW_IN",
            "VPD",
            "TA",
            "WTD",
            "WS",
            "PotRad",
            "PotRadHalfHourlyDiff",
            "PotRadDailyAvg",
            "PotRadDailyDiff",
            "WD_COS",
            "WD_SIN"
        ],
        "reco_inputs": [
            "DOY_sin",
            "DOY_cos",
            "TA",
            "TS_1",
            "TS_2",
            "TS_3",
            "TS_4",
            "WTD",
            "WS",
            "WD_COS",
            "WD_SIN",
            "NIGHTLY_NEE_AVG"
        ],
        "metrics": {
            "DT_GPP_vs_model": {
                "r2": 0.94,
                "rmse": 1.82
            },
            "NT_GPP_vs_model": {
                "r2": 0.92,
                "rmse": 2.01
            },
            "DT_RECO_vs_model": {
                "r2": 0.7,
                "rmse": 1.27
            },
            "NT_RECO_vs_model": {
                "r2": 0.64,
                "rmse": 1.22
            },
            "DT_GPP_vs_NT_GPP": {
                "r2": 0.95,
                "rmse": 1.55
            },
            "DT_RECO_vs_NT_RECO": {
                "r2": 0.85,
                "rmse": 1
            }
        }
    }
}
  
# print(dict)
for experiment_id in dict:
  experiment = dict[experiment_id]
  print(f"{experiment_id}\n"
        f"gpp inputs {'_'.join(experiment['gpp_inputs'])}\n"
        f"reco inputs {'_'.join(experiment['reco_inputs'])}\n"
        f"{experiment['metrics']['DT_GPP_vs_model']['r2']}\\t"
        f"{experiment['metrics']['NT_GPP_vs_model']['r2']}\\t"
        f"{experiment['metrics']['DT_RECO_vs_model']['r2']}\\t"
        f"{experiment['metrics']['NT_RECO_vs_model']['r2']}"
        f"\n"
        )
0.38,0.35,-4.67,-6.22  

