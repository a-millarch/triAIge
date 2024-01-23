# common
import sys
import pandas as pd
from matplotlib import pyplot as plt
import time
import argparse

# ml
from sklearn.metrics import roc_auc_score, roc_curve

from tsai.all import  Learner
from tsai.models.TabFusionTransformer import TSTabFusionTransformer
from tsai.callback.core import SaveModel, ShowGraphCallback2

from fastai.losses import CrossEntropyLossFlat, BCEWithLogitsLossFlat
from fastai.metrics import RocAucBinary, RocAucMulti

import mlflow
from mlflow import MlflowClient

# project imports
sys.path.append("..")
from src.features.fusion import FusionLoader
from src.models.ml_utils import get_class_weights
from src.models.modelcomponents import FocalLoss
from src.scripts.utils import load_configs, log_configs
from src.evaluation.metrics import get_metrics
# logging
import logging
from src.common.log_config import setup_logging, clear_log

clear_log()
setup_logging()
logger = logging.getLogger(__name__)

if __name__=="__main__":
    # options
    plt.rcParams["figure.figsize"] = (3,3)

    # from argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, help="optional target")
    parser.add_argument("--name", type=str, help="run name for experiment tracking")
    args = parser.parse_args()

    run_name= args.name

    ymls =ymls = {  "d": "data", 
                    "m": "model",
                    "e":"experiment"}

    cfg =load_configs(ymls)

    f = FusionLoader()
    if args.target: 
        f.from_cfg(cfg["d"], target=args.target)
    else: 
        logger.info("No target from argparse, using cfg")
        f.from_cfg(cfg["d"])
    logger.info(f"target: {f.target}")  
                   

    # Calculate weights, if weighted loss func
    weights= get_class_weights(f.full_tab_df, f.target)
    weights[1]= weights[1]*0.5

    # classes dict for tabular part
    classes = f.tab_dls.classes

    # https://github.com/timeseriesAI/tsai/blob/main/nbs/066_models.TabFusionTransformer.ipynb
    # if d_k, d_v, d_ff == None, then derived from d_model and n_heads

    model = TSTabFusionTransformer(f.dls.vars, f.dls.c, f.dls.len, classes, f.cont_names,
                                    fc_dropout=cfg["m"]["model"]["fc_dropout"],
                                    n_layers = cfg["m"]["model"]["n_layers"],
                                    n_heads = cfg["m"]["model"]["n_heads"],
                                    d_model=cfg["m"]["model"]["d_model"],
                                    # d_k=16, d_v = 16, d_ff = 512,
                                    
                                    )

    # Note, consider using TSlearner instead https://timeseriesai.github.io/tsai/tslearner.html
    learn = Learner(f.dls, model,
                    #loss_func=LabelSmoothingCrossEntropyFlat(),
                    #loss_func=FocalLoss(alpha = weights),
                    loss_func = CrossEntropyLossFlat(),
                    metrics=[RocAucBinary()],  
                    cbs=[SaveModel(monitor="valid_loss")]
                        )

    lr = learn.lr_find(show_plot=f.v)
    lr = lr.valley

    n_epochs= cfg["m"]["training"]["n_epochs"]

    mlflow.set_experiment(cfg["e"]["name"])
    with mlflow.start_run(run_name=f.target) as run:#run_name=run_name
        start = time.time()
        
        learn.fit_one_cycle(n_epochs, lr_max=lr)

        # logger
        elapsed = time.time() - start
        logger.info(f'Elapsed time: {elapsed}')
# add yaml params

        log_params={    "target":f.target, 
                        "max_sequence_length": cfg["d"]["data"]["cut_off_col_idx"],
                        "bin_frequency" : cfg["d"]["data"]["bin_freq"],
                        "classes":classes,
                        "sequential_fillna_mode": cfg["d"]["data"]["sequential_fillna_mode"],
                        "undersampling": cfg["d"]["data"]["undersampling"],
                        
                        "weights": weights,
                        "batch size":cfg["d"]["dls"]["bs"],
                        "learning rate": lr,

                        "fc_dropout":cfg["m"]["model"]["fc_dropout"]
                                            
                                            }
        for name, var in log_params.items():
            mlflow.log_param(name, var)
        log_configs(ymls)
        get_metrics(learn, f)
        mlflow.log_artifact("../logging/app.log")
        mlflow.log_artifact("../reports/figs/metric_plot.png")
        