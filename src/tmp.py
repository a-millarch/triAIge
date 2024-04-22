import sys
import os
import logging
import copy
import pandas as pd
import numpy as np

from tsai.inference import load_learner

from src.features.fusion import FusionLoader
from src.common.log_config import setup_logging, clear_log
from src.data.datasets import ppjDataset, TabularDataset

learner_path = "models/trained_model_learner.pkl"

from hydra import compose, initialize
with initialize(config_path='../configs', version_base="1.1"):
    cfg: dict = compose(config_name='default.yaml') #type: ignore


def colnames_from_cfg(cfg):
    # create new columns for each target, pred and actual y
    preds_list = ["preds_"+i for i in cfg.data.target]
    y_list = ["y_"+i for i in cfg.data.target]
    return preds_list, y_list


setup_logging()
logger = logging.getLogger(__name__)
clear_log()


def add_preds_to_tab(learn, f, dls, split):
    preds, ys = learn.get_preds(dl=dls)
    df = f.full_tab_df.iloc[split]

    logger.debug(f"length of preds: {len(preds)},\nlength of test_tab: {len(df)}")

    preds_list, y_list = colnames_from_cfg(cfg)

    df[preds_list] = preds
    df[y_list] = ys 
    return df

def save_pred_df(cfg=cfg):
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

    ds_tab = TabularDataset(base=base, default_mode= False)
    ds_tab.proces_categoricals()
    ds_tab.merge_categoricals()
    ds_tab.numeric_sequence_to_tabular()

    sets = {f.dls.train : f.splits[0], # training set
         f.dls.valid : f.splits[1], # valid set
         f.test_mixed_dls.valid : f.test_splits[1]} #test set
    
    # add predictions to train, valid, test and concat into one df with final population
    df_list = []
    for dls, split in sets.items():
        df_list.append(add_preds_to_tab(learn, f, dls, split))
    
    final_df = pd.concat(df_list)

    final_df.to_pickle('data/processed/predict_df.pkl')

if __name__=="__main__":
    save_pred_df()