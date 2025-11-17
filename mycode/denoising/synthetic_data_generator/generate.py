import neurokit2 as nk
import numpy as np
import yaml
from tqdm import tqdm


def get_synthetic_data(config_path: str) -> tuple:
    """Generate synthetic ECG data based on a given configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    tuple
        A tuple containing the simulated ECG signal and its processed information.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # get number of samples
    n_train = config["n_samples"]["train"]
    n_test = config["n_samples"]["test"]
    n_eval = config["n_samples"]["eval"]

    # initialize signal diversity parameters (inter record)
    std_ai = config["signal_diversity_parameters"]["std_ai"]
    std_bi = config["signal_diversity_parameters"]["std_bi"]
    mean_hr = config["baseline_parameters"]["heart_rate"]
    std_hr = config["signal_diversity_parameters"]["std_hr"]
    means_ai = (1.2, -5, 30, -7.5, 0.75)  # from paper
    means_bi = (0.25, 0.1, 0.1, 0.1, 0.4)  # from paper

    # generate records
    train_records = []
    test_records = []
    eval_records = []

    for n_samples, records in zip(
        [n_train, n_test, n_eval], [train_records, test_records, eval_records]
    ):
        for _ in tqdm(range(n_samples)):
            # initialize morphology parameters (intra record)
            ai = np.random.normal(loc=means_ai, scale=std_ai)
            bi = np.random.normal(loc=means_bi, scale=std_bi)
            heart_rate = np.random.normal(loc=mean_hr, scale=std_hr)
            duration = config["baseline_parameters"]["duration"] + 5
            cutoff = int(
                config["baseline_parameters"]["sampling_rate"] * 5
            )  # remove first 5 seconds

            # get ecg
            max_tries = 10
            for _ in range(
                max_tries
            ):  # some combinations of ai, bi, heart_rate may fail (rare)
                try:
                    ecg = nk.ecg_simulate(
                        duration=duration,
                        sampling_rate=config["baseline_parameters"]["sampling_rate"],
                        heart_rate=heart_rate,
                        heart_rate_std=config["signal_regularity_parameters"][
                            "heart_rate_std"
                        ],
                        lfhfratio=config["signal_regularity_parameters"]["lfhfratio"],
                        ai=ai,
                        bi=bi,
                    )
                    signals, _ = nk.ecg_process(
                        ecg,
                        sampling_rate=config["baseline_parameters"]["sampling_rate"],
                    )
                    clean_record = signals["ECG_Clean"].values[cutoff:]
                    records.append(clean_record)
                    break
                except Exception:
                    continue

    # convert to numpy arrays
    train_records = np.array(train_records)
    test_records = np.array(test_records)
    eval_records = np.array(eval_records)
    return train_records, test_records, eval_records
