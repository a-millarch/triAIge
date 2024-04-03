import sys
import os
import logging
import copy

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from tsai.inference import load_learner

sys.path.append("..")
import src
from src.features.fusion import FusionLoader
from src.common.log_config import setup_logging, clear_log
from src.data.datasets import ppjDataset, TabularDataset

####################################################################################
learner_path = "models/trained_model_learner.pkl"

from hydra import compose, initialize
with initialize(config_path='../configs', version_base="1.1"):
    cfg: dict = compose(config_name='default.yaml') #type: ignore

setup_logging()
logger = logging.getLogger(__name__)
clear_log()

###################################################################################
def colnames_from_cfg(cfg):
    # create new columns for each target, pred and actual y
    preds_list = ["preds_"+i for i in cfg.data.target]
    y_list = ["y_"+i for i in cfg.data.target]
    return preds_list, y_list

def check_learner_path():
    if os.path.exists(learner_path):
        logger.info("Predictions file found")
    else:
        logger.info("Generates predictions")
        save_preds()

def print_any(df):
    #logger.info(f'total: {len(df)}')
    logger.info(f'any preds sum:\n{df[["preds_any_binary", "y_any"]].sum()}')
    logger.info(f'false negatives:   {len(df[(df["y_any"] == 1) & (df["preds_any_binary"] != 1)])}')
    logger.info(f'false positives:   {len(df[(df["y_any"] != 1) & (df["preds_any_binary"] == 1)])}')

def add_binary_preds(thresholds, df):
    bin_cols=[]
    for col, thres in thresholds.items():
        newcol = f'{str(col)}_binary'
        df.loc[df[col]>=thres, newcol] = 1
        bin_cols.append(newcol)
    df.loc[df[bin_cols].eq(1).any(axis=1), "preds_any_binary"] = 1
    return df

def plot_confusion_matrices(df, outcome_names):
    num_outcomes = len(outcome_names)
    num_cols = 2  # Number of columns for subplots
    num_rows = (num_outcomes - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*5))

    for i, outcome_name in enumerate(outcome_names):
        row = i // num_cols
        col = i % num_cols

        y_true = df[f"y_{outcome_name}"].fillna(0)
        y_pred = df[f"preds_{outcome_name}_binary"].fillna(0)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])

        ax = axes[row, col] if num_rows > 1 else axes[col]
        disp.plot(ax=ax, cmap=plt.cm.Blues) #type: ignore
        ax.set_title(outcome_name)

    plt.tight_layout()
    plt.show()
############################################################################################
def save_preds(cfg = cfg):
    learn = load_learner(learner_path)
    logger.info("loaded trained model in learner object")
    base = ppjDataset(default_mode= False, max_na=0.9)
    base.collect_base_datasets(patients_info_file_name= cfg.data["patients_info_file_name"],
                                    ppj_file_name= cfg.data["ppj_file_name"])
    base.collect_subsets()
    base.clean_sequentials()
    base.sort_subsets()
    base.add_outcome()

    f = FusionLoader()
    f.from_cfg(cfg.data, base=copy.deepcopy(base))

    

    preds, ys = learn.get_preds(dl=f.test_mixed_dls.valid)
    df = f.full_tab_df.iloc[f.test_splits[1]]

    logger.debug(f"length of preds: {len(preds)},\nlength of test_tab: {len(df)}")
    
    preds_list, y_list = colnames_from_cfg(cfg)

    df[preds_list] = preds
    df[y_list] = ys 

    ds_tab = TabularDataset(base=base, default_mode= False)
    ds_tab.proces_categoricals()
    ds_tab.merge_categoricals()
    ds_tab.numeric_sequence_to_tabular()

    # criteria-based "predictions"
    all_tab = ds_tab.df[ds_tab.df.JournalID.isin(df.JournalID.unique())].copy(deep=True)
    all_tab.loc[(all_tab["M_Resp. frekvens_min"] < 10) | (all_tab["M_Resp. frekvens_max"] > 29), "RESP_CRITERIA"] = 1
    all_tab.loc[(all_tab["M_NInv Sys Blodtryk_min"] < 90) , "SBP_CRITERIA"] = 1
    df.loc[df["GCS"] < 14, "GCS_CRITERIA"] = 1
    df = df.merge(all_tab[["JournalID", "RESP_CRITERIA", "SBP_CRITERIA"]], how='left', on='JournalID')

    df.loc[df[y_list].eq(1).any(axis=1), "y_any"] = 1
    logger.info(f"Any y sum: {df.y_any.sum()}")

    logger.info(f'RESP_CRITERIA sum: {all_tab["RESP_CRITERIA"].sum()}\n'
            f'SBP_CRITERIA sum: {all_tab["SBP_CRITERIA"].sum()}\n'
            f'GCS_CRITERIA sum: {df["GCS_CRITERIA"].sum()}')

    df.to_pickle("data/processed/test_df_preds.pkl")
    logger.info("saved predictions")

def evaluate_preds(mode = cfg["evaluation"].mode, beta=cfg["evaluation"].beta):
    # setup
    check_learner_path()
    df = pd.read_pickle("data/processed/test_df_preds.pkl")
    logger.info(f"loaded test df, N={len(df)}")

    f_thresholds={}

    for col in cfg["data"].target:
        if mode == "f_beta":
            precision, recall, thresholds = precision_recall_curve(df["y_"+col], df["preds_"+col])
        # fscore = ((1+pow(beta,2)) * precision*recall)/(pow(beta,2)*precision+recall)
            fscore = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            # Find the optimal threshold
            index = np.nanargmax(fscore)
            thresholdOpt = round(thresholds[index], ndigits = 4)
            threshold = thresholdOpt

        elif mode == "f1":
            precision, recall, thresholds = precision_recall_curve(df["y_"+col], df["preds_"+col])
            fscore = (2 * precision * recall) / (precision + recall)  
            # Find the optimal threshold
            index = np.nanargmax(fscore)
            thresholdOpt = round(thresholds[index], ndigits = 4)
            threshold = thresholdOpt

        elif mode == "youden":
            fpr, tpr, thresholds = roc_curve(df["y_"+col], df["preds_"+col])
            idx = np.nanargmax(tpr - fpr)
            threshold = thresholds[idx]

        f_thresholds["preds_"+col] = threshold
    
    logger.info(f'thresholds: {f_thresholds}')
    new_df = add_binary_preds(f_thresholds, df.copy(deep=True))
    new_df.loc[new_df[["GCS_CRITERIA", "RESP_CRITERIA", "SBP_CRITERIA"]].eq(1).any(axis=1), "criteria_pred"] = 1

    new_df.loc[new_df[["preds_RH_plus_major_binary", "preds_neuro_major_binary"]].eq(1).any(axis=1), "preds_RHN_binary"] = 1
    new_df.loc[new_df[["y_RH_plus_major", "y_neuro_major"]].eq(1).any(axis=1), "y_RHN"] = 1

    logger.info(f'Removed due to criteria N = {new_df["criteria_pred"].sum()}, removed {len(new_df[new_df["criteria_pred"] ==1])} outcomes'
                ) #f'any preds sum: {new_df["preds_any_binary"].sum()}\n'
    test_df = new_df[new_df["criteria_pred"] != 1]
    logger.info(f'New N test cases: {len(test_df)}')
    #print_any(test_df)


    test_df.to_pickle("data/processed/evaluation_df.pkl")

    preds_list, y_list = colnames_from_cfg(cfg)
    for y,p in zip(y_list, preds_list):
        logger.info(f'{y: <25}: {test_df[y].sum()}')
        logger.info(f'{p: <25}: {test_df[p+"_binary"].sum()}\n')

    plot_outcome_names = ["RH_plus_major" , "neuro_major","abdominal_major" ,"vascular_major"]

    logger.info(f"all outcomes N={len(new_df)}")
    plot_confusion_matrices(new_df, plot_outcome_names)
    logger.info(f"non-criteria outcomes N={len(test_df)}")
    plot_confusion_matrices(test_df, plot_outcome_names)


###################################################################################

def main():
    pass
    #save_preds()
    #evaluate_preds()
        
if __name__=="__main__":
    main()