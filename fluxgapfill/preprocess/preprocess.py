import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
from joblib import Parallel, delayed

from .load import load_raw_data, load_test_data
from .artificial import learn_gap_dist, sample_artificial_gaps


def _preprocess_site(
        site, data_dir, args, na_values, split_method, dist,
        n_grid, n_mc, eval_frac, n_train, seed
):
    """Preprocess a single site. Called in parallel by preprocess()."""
    print(f"Data preprocessing... - site: {site}")

    site_data_dir = data_dir / site
    site_data_path = site_data_dir / "raw.csv"

    if site == 'TEST':
        site_data = load_test_data()
        if not os.path.exists(site_data_dir):
            os.makedirs(site_data_dir, exist_ok=True)
        site_data.to_csv(site_data_dir / 'raw.csv', index=False)
    else:
        site_data = load_raw_data(site_data_path, na_values)

    gap_indices = site_data.FCH4.isna()
    gap_set = site_data[gap_indices]

    if split_method == 'artificial':
        artificial_gap_dist = learn_gap_dist(
            flux_data=site_data.FCH4.values,
            dist=dist,
            n_grid=n_grid,
            n_mc=n_mc,
            seed=seed
        )

        test_flux = sample_artificial_gaps(
            flux_data=site_data.FCH4.values,
            sampling_pmf=artificial_gap_dist,
            eval_frac=eval_frac,
            seed=seed
        )
        test_indices = np.isnan(test_flux) & ~site_data.FCH4.isna()
        test_set = site_data[test_indices].reset_index(drop=True)
        trainval_set = site_data[~test_indices].reset_index(drop=True)

        train_valid_pairs = []
        for i in range(n_train):
            val_flux = sample_artificial_gaps(
                flux_data=trainval_set.FCH4.values,
                sampling_pmf=artificial_gap_dist,
                eval_frac=eval_frac,
                seed=seed + i
            )
            val_indices = np.isnan(val_flux) & ~trainval_set.FCH4.isna()
            val_set = trainval_set[val_indices]
            train_set = trainval_set[~val_indices & ~trainval_set.FCH4.isna()]
            train_valid_pairs.append((train_set, val_set))

    elif split_method == 'random':
        no_gap_set = site_data[~gap_indices]
        trainval_set, test_set = train_test_split(
            no_gap_set, test_size=eval_frac, random_state=seed
        )
        train_valid_pairs = []
        for i in range(n_train):
            train_set, val_set = train_test_split(
                trainval_set, test_size=eval_frac, random_state=seed + i
            )
            train_valid_pairs.append((train_set, val_set))

    else:
        raise ValueError(f'Splitting method {split_method} not supported.')

    for train_set, val_set in train_valid_pairs:
        assert site_data.shape[0] == (
            gap_set.shape[0] + test_set.shape[0] +
            val_set.shape[0] + train_set.shape[0]
        )

    train_data_dir = site_data_dir / "training"
    train_data_dir.mkdir(exist_ok=True)
    for i, (train_set, valid_set) in enumerate(train_valid_pairs):
        train_set.to_csv(train_data_dir / f"train{i+1}.csv", index=False)
        valid_set.to_csv(train_data_dir / f"valid{i+1}.csv", index=False)
    test_set.to_csv(site_data_dir / "test.csv", index=False)
    gap_set.to_csv(site_data_dir / "gap.csv", index=False)

    site_args_path = site_data_dir / "args.json"
    with site_args_path.open('w') as f:
        json.dump(args, f)

    print(f" - Done preprocessing data for site {site}.")
    print(f" - Processed data written to {site_data_dir}.\n")


def preprocess(
        data_dir,
        sites,
        na_values=-9999,
        split_method='artificial',
        dist='CramerVonMises',
        n_grid=10,
        n_mc=50,
        eval_frac=0.1,
        n_train=10,
        seed=1000,
        n_jobs=1,
        **kwargs
):
    """
    Preprocess the data and generate csvs to be input to the models.

    Assumes the site data is stored as a CSV in data/{SiteID}/raw.csv

    Args:
        data_dir (str): directory of the data folder containing all site folders.
        sites (str or list): Comma-separated list of site IDs to process.
                     Must match the name(s) of the data directories.
                     Examples:
                        - String input from command line: "siteA, siteB"
                        - List input from python program
                        - "TEST" loads testing data from Github and no local data required.
        na_values (int): Additional strings to recognize as NA/NaN. If dict passed,
                    specific per-column NA values. By default the following values are
                    interpreted as NaN: '', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                    '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
                    'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'.
        split_method (str): How to split the data into training, validation,
                            and test sets.
                            Options: ['artificial', 'random']
        dist (str): Distance measure to use for evaluating the similarity
                    between the empirical and approximated gap distribution.
                    Options: ['CramerVonMises', 'KolmogorovSmirnoff',
                              'ChiSquare', 'HistogramIntersection',
                              'Hellinger']
        n_grid (int): Width of the grid search per hyperparameter.
                      Must be a positive integer.
        n_mc (int): Number of Monte Carlo interations for estimating
                    the artificial gap distribution.
                    Must be a positive integer.
        eval_frac (float): Proportion of the data to use for testing and validation.
                           Must be a float between 0 and 1.
        n_train (int): Number of paired training and validation sets to
                       generate.
                       Must be a positive integer.
        seed (int): Random seed to initialize pseudorandom number generator.
        n_jobs (int): Number of parallel jobs for multi-site processing.
                      -1 uses all available CPU cores. Default: 1 (sequential).

    Writes all preprocessed data to data/{SiteID}/ for each {SiteID}
    """
    args = locals().copy()
    data_dir = Path(data_dir)

    if isinstance(sites, str):
        sites = [s.strip() for s in sites.split(",")]

    # n_jobs=1: sequential (safe default); -1 or >1: parallel across sites
    # Note: when n_jobs != 1, learn_gap_dist's internal joblib parallelism
    # still runs, so total workers = n_sites * n_cpus. Set n_jobs wisely.
    Parallel(n_jobs=n_jobs)(
        delayed(_preprocess_site)(
            site, data_dir, args, na_values, split_method, dist,
            n_grid, n_mc, eval_frac, n_train, seed
        )
        for site in sites
    )
