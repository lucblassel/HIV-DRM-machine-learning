import argparse
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

TASKS = {
    "results_treatment_all_features_cleaned": "all features\nkept",
    "results_treatment_no_DRMs_cleaned": "DRM features\nremoved",
    "results_treatment_no_DRMs_no_seqs_cleaned": "DRM features &\nDRM sequences\nremoved",
    "results_whole_training_set": "all features\nkept",
    "results_whole_training_set_no_DRMs": "DRM features\nremoved",
    "results_whole_training_set_no_DRMs_no_seqs": "DRM features &\nDRM sequences\nremoved",
}

MODELS = {
    "Bayes": "NB",
    "Logistic": "LR",
    "FisherBonf1": "EFT",
    "FisherBonf2": "B2",
    "FisherBH1": "BH1",
    "FisherBH2": "BH2",
    "DRMs1": "D1",
    "DRMs2": "D2",
    "SDRMs1": "SD1",
    "SDRMs2": "SD2",
}

DATASETS = ["UK", "Africa"]

GROUPER = ["task", "subtype", "dataset", "shorthand"]

REGEX = {
    "dir": r"(?P<dir>[^/]+)/(?P<subtype>[^/]+)/(?P<target>.*)",
    "pred": (
        r"final-(?P<model>[^-]+)-(?P<dataset>[^-]+)"
        r"-predictions-num(?P<num>\d+).tsv$"
    ),
    "coef": r"final-(?P<model>[^-]+)-coefficients-num(?P<num>\d+).tsv",
}


def get_paths(dirs, mode="pred"):
    """
    Generates relative paths to the result files from the top level directories
    """
    regex = REGEX.get(mode)
    for d in dirs:
        for root, _, filenames in os.walk(d):
            for filename in filenames:
                file_match = re.match(regex, filename)
                if file_match is not None:
                    meta = {
                        **re.match(REGEX["dir"], root).groupdict(),
                        **file_match.groupdict(),
                    }
                    yield os.path.join(root, filename), meta


def get_meta(metadata_files):
    """
    Reads and processes specified metadata files
    """
    metas = []
    for filename in metadata_files:
        dataset = filename.split("_")[0].capitalize()
        meta = pd.read_csv(filename, sep="\t", index_col=0, header=0).rename(
            {"REGAsubtype": "subtype"}, axis=1
        )
        meta["dataset"] = {"Uk": "UK"}.get(dataset, dataset)
        metas.append(meta)

    all_meta = pd.concat(metas, axis=0, sort=False).rename(
        {"subtype": "sample_subtype"}, axis=1
    )
    all_meta.index = all_meta.index.astype(str).rename("index")

    return all_meta


def remove_training_subtype(preds_df):
    """
    Removes predicitons on samples of subtype identical to the training subtype
    """
    df = preds_df.copy()
    df["subtype_list"] = df["sample_subtype"].apply(lambda x: x.split())
    df["different"] = df.apply(lambda x: x["subtype"] not in x["subtype_list"], axis=1)
    different = df[df["different"]].drop(["different", "subtype_list"], axis=1)

    different["dataset"] += " different"

    return different


def remove_DRM_sequences(dfs):
    """
    removes predicitons on samples that have at least on known DRM
    """
    removed = []
    for df in dfs:
        sub = df[df["task"] == "DRM features &\nDRM sequences\nremoved"]
        sub = sub[sub["hasDRM"] == 0]
        sub["dataset"] += " no test DRM"
        removed.append(sub)

    return pd.concat(removed, axis=0, sort=False)


def get_preds(dirs):
    """
    Reads and concatenates predictions by all trained classifiers
    """
    preds = []
    paths = list(get_paths(dirs, mode="pred"))

    for path, meta in tqdm(paths, desc="predictions"):
        df = pd.read_csv(path, sep="\t", index_col=0)
        for k, v in meta.items():
            df[k] = v
        preds.append(df)

    preds_df = pd.concat(preds, axis=0, sort=False)
    preds_df["task"] = preds_df["dir"].apply(lambda x: TASKS.get(x, x))
    preds_df["shorthand"] = preds_df["model"].apply(lambda x: MODELS.get(x, x))
    preds_df["dataset"] = preds_df["dataset"].apply(
        lambda x: {"test": "UK"}.get(x, x.capitalize())
    )
    preds_df.index = preds_df.index.astype(str).rename("index")
    return preds_df


def get_coefs(dirs):
    """
    Reads and concatenates model weights / feature importances for all trained classifiers
    """
    coefs = []
    coef_paths = list(get_paths(dirs, mode="coef"))

    for path, meta in tqdm(coef_paths, desc="coeficients"):
        df = pd.read_csv(path, sep="\t", index_col=0)

        if meta["model"] == "Bayes":
            df = df.applymap(np.exp)

        df["normalized"] = False

        for k, v in meta.items():
            df[k] = v

        coefs.append(df)

    return (
        pd.concat(coefs, axis=0)
        .fillna(0)
        .groupby(["dir", "model", "subtype", "target", "normalized"])
        .mean()
        .reset_index()
    )


def group_preds(preds_df):
    """
    Gets majority prediction for models with repeated training
    """
    df = preds_df.copy()
    grouper = ["index", "task", "subtype", "shorthand", "real", "dataset", "target"]
    grouped = (
        df.reset_index()
        .groupby(grouper)["pred"]
        .mean()
        .apply(round)
        .reset_index()
        .set_index("index")
    )
    return grouped


def get_args():
    """
    Gets command line arguments
    """
    arg_parser = argparse.ArgumentParser(
        description=(
            "This script takes results generated by one or several runs of the Snakefile_main.smk pipeline "
            "and outputs concatenated tab delimited files with classifier predictions and feature weights."
        )
    )
    arg_parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="top level directories, containing the results, generated by the main pipeline.",
    )
    arg_parser.add_argument(
        "--metadata",
        nargs="+",
        required=True,
        help="list of metadata files for all training and testing sets used to generate the results.",
    )
    arg_parser.add_argument(
        "--result_dir",
        required=True,
        help="directory in whic to output the concatenated result files.",
    )
    return arg_parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.result_dir, exist_ok=True)

    metadata = get_meta(args.metadata)
    preds = group_preds(get_preds(args.dirs)).join(
        metadata[["hasDRM", "sample_subtype"]]
    )
    different = remove_training_subtype(preds)
    removed = remove_DRM_sequences([preds, different])
    all_preds = pd.concat([preds, different, removed])
    coefs_df = get_coefs(args.dirs)
    all_preds.to_csv(
        os.path.join(args.result_dir, "all_preds.tsv"),
        sep="\t",
        header=True,
        index=True,
    )
    coefs_df.to_csv(
        os.path.join(args.result_dir, "coefs_df.tsv"),
        sep="\t",
        header=True,
        index=True,
    )


if __name__ == "__main__":
    main()
