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
from tsai.models.TabFusionTransformer import TSTabFusionTransformer
from tsai.callback.core import SaveModel, ShowGraphCallback2

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
#from src.custom.tsai_custom import TrainingShowGraph
#from src.custom.custom_fusion_model import *


import hydra
from omegaconf import DictConfig

from src.common.log_config import setup_logging, clear_log


@hydra.main(version_base=None, config_path="../../configs", config_name="default.yaml")
def train_main(cfg:DictConfig):
    setup_logging()
    logger = logging.getLogger(__name__)

    clear_log()
 
    logger.info(f"{cfg.data.target}")

    logger.info("Creating ppjDataset/base")

    # create base dataset
    base = ppjDataset(default_mode= False, max_na=0.9)
    base.collect_base_datasets(patients_info_file_name= cfg.data["patients_info_file_name"],
                                    ppj_file_name= cfg.data["ppj_file_name"])

    base.collect_subsets()
    base.clean_sequentials()
    base.sort_subsets()
    base.add_outcome()

    
if __name__=="__main__":
    train_main()