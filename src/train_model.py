import sys
import time
import logging
import importlib
import copy

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tsai.all import  Learner
from tsai.models.TabFusionTransformer import TSTabFusionTransformer, TabFusionTransformer
from tsai.callback.core import SaveModel, ShowGraphCallback2, EarlyStoppingCallback

from fastai.losses import CrossEntropyLossFlat,BCEWithLogitsLossFlat
from fastai.metrics import RocAucBinary,RocAucMulti, APScoreMulti

import mlflow
from mlflow import MlflowClient

sys.path.append("..")
import src
from src.data.datasets import ppjDataset
from src.features.fusion import FusionLoader
from src.visualization.visualize import plot_evaluation
from src.models.ml_utils import get_class_weights
from src.models.modelcomponents import FocalLoss
from src.scripts.utils import load_configs, log_configs
from src.evaluation.metrics import get_metrics, multilabel_roc_analysis_and_plot, multilabel_roc_pr_analysis_and_plot
from src.custom.tsai_custom import TrainingShowGraph, save_loss_plot
#from src.custom.custom_fusion_model import *

import hydra
from omegaconf import DictConfig

from src.common.log_config import setup_logging, clear_log

@hydra.main(version_base=None, config_path="../configs", config_name="default.yaml")
def train_main(cfg:DictConfig):
    setup_logging()
    logger = logging.getLogger(__name__)
    clear_log()
    # Load and proces data
    logger.info(f"{cfg.data.target}")
    logger.info("Creating ppjDataset/base")
    base = ppjDataset(default_mode= False, max_na=0.9)
    base.collect_base_datasets(patients_info_file_name= cfg.data["patients_info_file_name"],
                                    ppj_file_name= cfg.data["ppj_file_name"])

    base.collect_subsets()
    base.clean_sequentials()
    base.sort_subsets()
    base.add_outcome()

    f = FusionLoader()
    f.from_cfg(cfg.data, base=copy.deepcopy(base))

    # Calculate weights, if weighted loss func
    ws = []
    for targ in f.target: #type: ignore
        cw = get_class_weights(f.full_tab_df, targ)
    # cw = cw * 
        ws.append(cw[1])
    ws = torch.tensor(np.array(ws))
    ws = torch.mul(ws, cfg.model["weights_factor"])

    # Set loss func
    if cfg.model["weights"]:
        logger.info(f"Using class weights: {ws}")
        loss_func = BCEWithLogitsLossFlat(pos_weight=ws)
    else:
        loss_func = BCEWithLogitsLossFlat()

    # Create model
    # https://github.com/timeseriesAI/tsai/blob/main/nbs/066_models.TabFusionTransformer.ipynb
    
    classes = f.tab_dls.classes #classes dict for tabular part of model
    model = TSTabFusionTransformer(f.dls.vars, len(f.target), f.dls.len, classes, f.cont_names, # type: ignore
                                    fc_dropout=cfg["model"]["fc_dropout"],
                                    n_layers = cfg["model"]["n_layers"],
                                    n_heads = cfg["model"]["n_heads"], # if d_k, d_v, d_ff == None, then derived from d_model and n_heads
                                    d_model=cfg["model"]["d_model"],
                                    res_dropout = cfg["model"]["res_dropout"]
                                    )
    
    # Create learner
    #escb = EarlyStoppingCallback(monitor='valid_loss', patience=3)
    smcb =SaveModel(monitor="valid_loss", fname = "model", 
                                   every_epoch=False, at_end=False, with_opt=False, reset_on_fit=True,comp=None)

    learn = Learner(f.dls, model,
                    loss_func =  loss_func,
                    metrics=[RocAucMulti(average='macro'), RocAucMulti(average=None)], #type: ignore
                    cbs=[smcb]
                        )
    
    # set learning rate from fastai lr-finder by hpm factor (most often reduced)
    lr = learn.lr_find(show_plot=False).valley * cfg["model"]["lr_factor"]

    # Prepare for training cycle
    mlflow.set_experiment(cfg.experiment["name"] )
    with mlflow.start_run(run_name=cfg.experiment["run_name"], tags = {"note":cfg.experiment["note"]}) as run:
        start = time.time()
        
        learn.fit_one_cycle(cfg["model"]["n_epochs"], lr_max=lr)
        learn.export("models/trained_model_learner.pkl")
        # logging
        elapsed = time.time() - start
        logger.info(f'Elapsed time: {elapsed}')

        # Add parameters to Mlflow logging
        log_params={    "max_sequence_length": cfg["data"]["cut_off_col_idx"],
                        "bin_frequency" : cfg["data"]["bin_freq"],
                        "sequential_fillna_mode": cfg["data"]["sequential_fillna_mode"],

                        "batch size":cfg["data"]["bs"],
                        "learning rate": lr,

                        "fc_dropout":cfg["model"]["fc_dropout"],
                        "n_heads":cfg["model"]["n_heads"],
                        "n_layers":cfg["model"]["n_layers"],
                        "d_model":cfg["model"]["d_model"]
                                            
                                            }
        for name, var in log_params.items():
            mlflow.log_param(name, var)
            
        mlflow.log_params(cfg["data"]) # type: ignore
        mlflow.log_params(cfg["model"]) # type: ignore
     
   
        # Plot and log losses from traning cycle
        save_loss_plot(learn, "reports/figures/loss_plot.png")

        # Evaluate on test dataset and log metrics
        rauc, prauc = multilabel_roc_pr_analysis_and_plot(learn, f.target, dl=f.test_mixed_dls.valid, #type: ignore
                                                          save_path='reports/figures/test_metric_plot.png', show=cfg["experiment"]["show_plots"])  

        for name, score in zip(f.target, rauc):#type: ignore
            mlflow.log_metric(f"test_roc_auc_{name}", score)

        for name, score in zip(f.target, prauc):#type: ignore
            mlflow.log_metric(f"test_pr_auc_{name}", score)

        # Get validation plot
        _, _ = multilabel_roc_pr_analysis_and_plot(learn, f.target, dl=None, #type: ignore - if none, then using validation set
                                                          save_path='reports/figures/valid_metric_plot.png', show=cfg["experiment"]["show_plots"]) 
        # Add artifacts to mlflow
        artifact_paths = [  "logging/app.log",
                            'reports/figures/test_metric_plot.png',
                            'reports/figures/valid_metric_plot.png',
                            "reports/figures/loss_plot.png",
                            "models/model.pth",
                            "models/trained_model_learner.pkl"
                            ]
        for artifact in artifact_paths:
            mlflow.log_artifact(artifact)

    return rauc[0] #test roc_auc for any 
    #return learn.final_record[2][0]
     
    
if __name__=="__main__":
    train_main()