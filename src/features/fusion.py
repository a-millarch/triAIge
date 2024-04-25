import pandas as pd 
import numpy as np 
import pickle 

from src.data.datasets import TimeSeriesDataset, TabularDataset, ppjDataset, DTD
from src.custom.tsai_custom import CustomTSMultiLabelClassification
from src.features.loader import sks_labels


from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import TSStandardize, TSNormalize, TSMultiLabelClassification
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.validation import get_splits
from tsai.data.preparation import df2xy
from tsai.all import TensorMultiCategory

from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize
from omegaconf import OmegaConf

from pathlib import Path
p = Path(__file__).parents[2]

import logging
from src.common.log_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


class FusionLoader():
    def __init__(self, default_mode=True, verbose=False):
        if default_mode:
            pass
        else:pass
        # todo: do more stuff in case no cfg...
        self.v=verbose 


    def from_cfg(self, cfg,
                 base: ppjDataset | None = None,
                 target: str | list | None = None):
        """ instantiate from config file """
        
        self.cfg = cfg    
        self.v = cfg["verbose"]

        if target:
            self.target = target
        else:
            self.target = cfg["target"]
            self.target = OmegaConf.to_object(self.target)

        if isinstance(self.target, str):
            self.multilabel = False
        else:
            self.multilabel = True 

        if base:
            self.base = base
        else:
            logger.info("No base file, collecting")
            self.collect_base(patients_info_file_name= cfg["patients_info_file_name"],
                                ppj_file_name= cfg["ppj_file_name"])
            
        self.get_dfs(   self.target, 
                        outcome_creation_mode = cfg["outcome_creation_mode"], 
                        cut_off_col_idx = cfg["cut_off_col_idx"],
                        bin_freq= cfg["bin_freq"],  # type: ignore
                        fillmode=  cfg["sequential_fillna_mode"],
                        undersampling = cfg["undersampling"],
                        upper_seq_limit = cfg["upper_seq_limit"],
                        dtd_keep_colnames = cfg["dtd_keep_colnames"],
                        min_seq_len = cfg["min_seq_len"],
                        bin_operation=cfg["bin_operation"])
        
        self.get_splits_from_dfs()
        self.get_dls()
        self.get_test_dls()

    def collect_base(self,
                     patients_info_file_name = None, ppj_file_name=None  ):
        """ Collect base dataset (ppjDataset) either from pkl or construct 

        Not properly implemented...
        """
        logger.info("Creating ppjDataset/base")
        base = ppjDataset(default_mode= False, max_na=0.9)
        base.collect_base_datasets( patients_info_file_name=patients_info_file_name,  # type: ignore
                                    ppj_file_name= ppj_file_name) # type: ignore
        

        base.collect_subsets()
        base.clean_sequentials()
        base.sort_subsets()
        base.add_outcome()

        self.base = base

    def get_dfs(self, target, 
                cut_off_col_idx=20, 
                bin_freq ="30S", 
                outcome_creation_mode= "categorical_procedure", 
                fillmode ="mean",
                undersampling=0.3,
                upper_seq_limit = 100,
                dtd_keep_colnames = None,
                min_seq_len = 5,
                bin_operation ="max",
                ):
        
        """ Collect timeseriesdataset obj and tabulardataset object"""
        base = self.base

        ds_tab = TabularDataset(base=base, default_mode= False)
        ds_tab.proces_categoricals()
        ds_tab.merge_categoricals()

        ds = TimeSeriesDataset(base = base, target=target)
        ds.compute(cutoff_col_idx=cut_off_col_idx, 
                   undersampling=undersampling, 
                   bin_freq=bin_freq, 
                   bin_operation=bin_operation,
                   fillmode = fillmode,
                   upper_seq_limit = upper_seq_limit,
                   min_seq_len = min_seq_len)

        # Keep lowest GCS value, drop duplicates
        tmp_GCS = ds_tab.base.subset_numericals["GCS"].sort_values(by=["JournalID", "Value"], ascending=True).drop_duplicates(subset="JournalID",keep='first') # type: ignore
        tmp_GCS.rename(columns={"Value":"GCS"}, inplace=True)
        ds_tab.df = ds_tab.df.merge(tmp_GCS[["JournalID", "GCS"]], on ="JournalID", how="left")



        # add age (move this to ppj patient info at some point)
        ds_tab.df["age"] = ((pd.to_datetime(ds_tab.df.ServiceDate) - pd.to_datetime(ds_tab.df.Fødselsdato)).dt.days/365).round(0)
  
        # define columns (should rename in PPJdataset as well)
        keep_cat_cols= ["Køn",]
        keep_cont_cols= ["age", "GCS" ]

        # DTD
        if dtd_keep_colnames is not None:
            # Collect DTD and filter by categories of column names
            dtd = DTD()
            dtd_df = dtd.get_by_colnames(dtd_keep_colnames) #type: ignore 
            logger.info(f"Adding DTD info for {len(dtd_df)} trajectories out of {len(ds_tab.df)} trajectories")

            # adjust cont and cat names according to additions from DTD
            if isinstance(dtd_keep_colnames, str):
                if dtd_keep_colnames in dtd.colnames_types["cat"]:
                        keep_cat_cols = keep_cat_cols + dtd.colnames[dtd_keep_colnames]
                else:
                    keep_cont_cols = keep_cont_cols + dtd.colnames[dtd_keep_colnames]

            else:
                for c in dtd_keep_colnames:
                    if c in dtd.colnames_types["cat"]:
                        keep_cat_cols = keep_cat_cols + dtd.colnames[c]
                    elif c in dtd.colnames_types["cont"]:
                        keep_cont_cols = keep_cont_cols + dtd.colnames[c]
                    else: pass
            # merge dataframe onto tab df NOTE: left merge for now        
            ds_tab.df = ds_tab.df.merge(dtd_df, how='left', on="JournalID")

        cat_names=  keep_cat_cols+ds_tab.base.cat_names
        #cat_names = ds_tab.base.cat_names
        cont_names = keep_cont_cols 

        if self.multilabel:
            keep_cols = ["JournalID",] +target + cat_names+  cont_names
        else:
            keep_cols = ["JournalID", target] + cat_names+  cont_names

        self.cont_names = cont_names.copy()
        self.cat_names = cat_names.copy()

        # reduce df
# REDUCING TIME HERE
        ds_tab.df = ds_tab.df[pd.to_datetime(ds_tab.df.ServiceDate).dt.year > 2016]       
        tab_df = ds_tab.df[ keep_cols].copy(deep=True)
        logger.debug(f"columns: {tab_df.columns}")
        tab_df.drop_duplicates(inplace=True)
        logger.info(f"Tabular dataset length before reducing: {len(tab_df)}")
 # HACKY SHIT
        tab_df.loc[0, self.cont_names+ self.cat_names] = np.nan #type ignore
        #tab_df = tab_df[(tab_df.age.notnull())&  (tab_df.Køn.notnull())]
        # 3, 90
        tab_df = tab_df[(tab_df.age.notnull())& (tab_df.age.between(3,90, inclusive="both")) &  (tab_df.Køn.notnull())]
        #tab_df["age"].fillna(64, inplace=True)
        #tab_df = tab_df[tab_df.]
        
# END

        # reduce patient mapping and merge
        ds.patient_map = ds.patient_map[ds.patient_map["sample"].isin(ds.long_df["sample"].unique())]
        tab_df = tab_df.merge(ds.patient_map, on="JournalID", how="inner")
        tab_df.sort_values(by="sample", inplace=True)
        tab_df.reset_index(drop=True, inplace=True)
   
        # set copy to avoid changes in origin long df using df2xy
        ldf = ds.long_df.copy(deep=True)
    # MORE HACKY S
        ldf = ldf[ldf["sample"].isin(tab_df["sample"].unique())]
        logger.info(f"Tabular dataset length after reducing: {len(tab_df)}")

        if self.multilabel:
            def y_func(o): return o
            target_col_len = len(target)
        else:
            def y_func(o): return o[:,0]
            target_col_len = 1

        self.X, self.y = df2xy(ldf, sample_col='sample', feat_col='feature', 
                               target_col=target, data_cols=ldf.columns[2:-target_col_len],
                               y_func=y_func)
        
        # if a list, overwrite the y with my own structure
        if self.multilabel:
            logger.info("Replacing fusion.y")
            self.y = ldf.drop_duplicates(subset="sample")[self.target].values.tolist() 
        else: pass


        # final dataframes set to self
        self.ldf = ldf.copy(deep=True)
        self.timeseries_ds = ds
        self.tab_ds = ds_tab
        
        self.full_tab_df = tab_df
        self.tdf = tab_df.drop(columns=["JournalID", "sample"]).copy(deep=True)

    def get_splits_from_dfs(self):
        """ uses TSAI lib get_splits and set train/val splits and test splits
        """
        target = self.target
        if self.multilabel:
            ts_splits = get_splits(np.array([l[0] for l in self.y]) , valid_size=self.cfg["valid_size"], test_size=self.cfg["test_size"]  ,stratify=True, random_state=self.cfg["seed"], shuffle=True, check_splits= True, show_plot=self.v)

        else:
            ts_splits = get_splits(self.y, valid_size=self.cfg["valid_size"], test_size=self.cfg["test_size"]  ,stratify=True, random_state=self.cfg["seed"], shuffle=True, check_splits= True, show_plot=self.v)

        logger.info(f"{len(self.full_tab_df['sample'].unique())} trajectories in total with {self.full_tab_df[target].sum()} positive outcomes")
        
        valid_idxs = list(ts_splits[1])
        n_val_idxs = len(valid_idxs)
        #logger.info(f"{n_val_idxs} in valid dataset with {self.y[valid_idxs].sum()} positive outcomes")
        
        test_idxs = list(ts_splits[2])
        n_test_idxs = len(test_idxs)
        #logger.info(f"{n_test_idxs} in test dataset with {self.y[test_idxs].sum()} positive outcomes")

        # define new splits 
        self.splits = ((ts_splits[0],)+ (ts_splits[1],))

        # share training data, treating test as valid
        self.test_splits = ((ts_splits[0],)+  (ts_splits[2],))

    def get_dls(self):
        """ Collect tab and ts dataloaders to create (mixed) dls
        """
        #dataloaders
        procs = [Categorify, FillMissing, Normalize]
        tab_dls = get_tabular_dls(self.tdf, procs=procs, 
                                  cat_names=self.cat_names.copy(), cont_names=self.cont_names.copy(), 
                                  y_names= self.target, 
                                  splits= self.splits)
        
        if self.multilabel:
            tfms  = [None, CustomTSMultiLabelClassification()]
            #tfms  = [None, TSMultiLabelClassification()] # TSMultiLabelClassification() == [MultiCategorize(), OneHotEncode()]        
        else:
            tfms  = [None, [Categorize()]]

        #batch_tfms=TSNormalize(by_var=True, range=(0,1))
            
        #batch_tfms=TSStandardize(by_var=True, verbose=True)
        batch_tfms =TSNormalize(by_var=True, range=(-1,1))

        ts_dls = get_ts_dls(self.X, self.y, splits=self.splits, tfms=tfms, batch_tfms=batch_tfms)

        mixed_dls = get_mixed_dls( ts_dls, tab_dls, bs=self.cfg["bs"])
        #if self.v:
        #    mixed_dls.show_batch()

        self.ts_dls = ts_dls
        self.tab_dls = tab_dls
        self.dls = mixed_dls

    def get_test_dls(self):
        """ test dataloaders using test splits
        """
        # create tab dataloader
        procs = [Categorify, FillMissing, Normalize]

        self.test_tab_dls = get_tabular_dls(self.tdf
                                        , procs=procs
                                        , cat_names=self.cat_names.copy()
                                        , cont_names=self.cont_names.copy()
                                        , y_names= self.target
                                        , splits= self.test_splits
                                    )
        
                # ts dataloader
                
        if self.multilabel:
            tfms  = [None, CustomTSMultiLabelClassification()]     
        else:
            tfms  = [None, [Categorize()]]

        
        #batch_tfms=TSStandardize(by_var=True, verbose=True)
        batch_tfms =TSNormalize(by_var=True, range=(-1,1))
        
        #test_ts_dls = get_ts_dls(self.X, self.y, splits=self.test_splits, tfms=tfms, batch_tfms=TSStandardize(by_var=True))
        self.test_ts_dls = get_ts_dls(self.X, self.y, splits=self.test_splits, tfms=tfms, batch_tfms=batch_tfms)
        # mix
        self.test_mixed_dls = get_mixed_dls( self.test_ts_dls, self.test_tab_dls)