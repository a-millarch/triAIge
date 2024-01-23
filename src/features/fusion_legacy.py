import pandas as pd 

from src.data.datasets import TimeSeriesDataset, TabularDataset, ppjDataset

from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import TSStandardize, TSNormalize
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.validation import get_splits
from tsai.data.preparation import df2xy

from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize

import logging
from pathlib import Path

p = Path(__file__).parents[2]
logging.basicConfig(level = logging.INFO, filename=p.joinpath('logging/app.log'), 
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

class FusionLoader():
    def __init__(self, default_mode=True, verbose=False):
        if default_mode:
            pass
        else:pass
        # todo: do more stuff in case no cfg...
        self.v=verbose 

    def from_cfg(self, cfg, target=None):
        """ instantiate from config file """

        if target:
            self.target = target
        else:
            self.target = cfg["common"]["target"]
        self.cfg = cfg    
        self.v = cfg["common"]["verbose"]

        self.get_dfs(   self.target, 
                        outcome_creation_mode = cfg["data"]["outcome_creation_mode"], 
                        cut_off_col_idx = cfg["data"]["cut_off_col_idx"],
                        bin_freq= cfg["data"]["bin_freq"],  # type: ignore
                        fillmode=  cfg["data"]["sequential_fillna_mode"])
        
        self.get_splits_from_dfs()
        self.get_dls()
        self.get_test_dls()

    def get_dfs(self, target, 
                cut_off_col_idx=20, 
                bin_freq ="30S", 
                outcome_creation_mode= "categorical_procedure", 
                fillmode ="mean"):
        
        """ Collect timeseriesdataset obj and tabulardataset object"""

        ds = TimeSeriesDataset(target=target)
        ds.compute(cutoff_col_idx=cut_off_col_idx, remove_short_nulls=True, 
                   outcome_mode =outcome_creation_mode, bin_freq=bin_freq, fillmode = fillmode)

        ds_tab = TabularDataset(default_mode= False)
        # refactor numeric_seq2tab - separate ds.df maker.
        ds_tab.proces_categoricals()
        ds_tab.merge_categoricals() 

        ds_tab.add_outcome(outcome_mode =outcome_creation_mode)

        # Keep lowest GCS value, drop duplicates
        tmp_GCS = ds_tab.subset_numericals["GCS"].sort_values(by=["JournalID", "Value"], ascending=True).drop_duplicates(subset="JournalID",keep='first') # type: ignore
        tmp_GCS.rename(columns={"Value":"GCS"}, inplace=True)
        ds_tab.df = ds_tab.df.merge(tmp_GCS[["JournalID", "GCS"]], on ="JournalID", how="left")

        # add age (move this to ppj patient info at some point)
        ds_tab.df["age"] = ((pd.to_datetime(ds_tab.df.ServiceDate) - pd.to_datetime(ds_tab.df.Fødselsdato)).dt.days/365).round(0)
        ds_tab.df = ds_tab.df.merge(ds_tab.patients_info[["JournalID", target]], how='left', on = "JournalID")

        # define columns (should rename in PPJdataset as well)
        keep_cat_cols= ["Køn",]
        keep_cont_cols= ["age", "GCS" ]

        cat_names=  keep_cat_cols+ds_tab.cat_names
        cont_names = keep_cont_cols 

        keep_cols = ["JournalID", target] + cat_names+  cont_names
        self.cont_names = cont_names.copy()
        self.cat_names = cat_names.copy()
        # reduce df
        tab_df = ds_tab.df[ keep_cols].copy(deep=True)
        tab_df.drop_duplicates(inplace=True)
        # reduce patient mapping and merge
        ds.patient_map = ds.patient_map[ds.patient_map["sample"].isin(ds.long_df["sample"].unique())]
        tab_df = tab_df.merge(ds.patient_map, on="JournalID", how="inner")
        tab_df.sort_values(by="sample", inplace=True)
        tab_df.reset_index(drop=True, inplace=True)
        # set copy to avoid changes in origin long df using df2xy
        ldf = ds.long_df.copy(deep=True)
        def y_func(o): return o[:,0]
        self.X, self.y = df2xy(ldf, sample_col='sample', feat_col='feature', 
                               target_col=target, data_cols=ldf.columns[2:-1],
                               y_func=y_func)

        # final dataframes set to self
        self.ldf = ldf
        self.timeseries_ds = ds
        self.tab_ds = ds_tab
        
        self.full_tab_df = tab_df
        self.tdf = tab_df.drop(columns=["JournalID", "sample"]).copy(deep=True)

    def get_splits_from_dfs(self):
        """ uses TSAI lib get_splits and set train/val splits and test splits
        """
        target = self.target

        ts_splits = get_splits(self.y, valid_size=self.cfg["preproces"]["valid_size"], test_size=self.cfg["preproces"]["test_size"]  ,stratify=True, random_state=self.cfg["common"]["seed"], shuffle=True, check_splits= True, show_plot=self.v)

        logging.info(f"{len(self.full_tab_df['sample'].unique())} trajectories in total with {self.full_tab_df[target].sum()} positive outcomes")
        
        valid_idxs = list(ts_splits[1])
        n_val_idxs = len(valid_idxs)
        logging.info(f"{n_val_idxs} in valid dataset with {self.y[valid_idxs].sum()} positive outcomes")
        
        test_idxs = list(ts_splits[2])
        n_test_idxs = len(test_idxs)
        logging.info(f"{n_test_idxs} in test dataset with {self.y[test_idxs].sum()} positive outcomes")

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
        tfms  = [None, [Categorize()]]
        #ts_dls = get_ts_dls(self.X, self.y, splits=self.splits, tfms=tfms, batch_tfms=TSStandardize(by_var=True))
        ts_dls = get_ts_dls(self.X, self.y, splits=self.splits, tfms=tfms, batch_tfms=TSNormalize(by_var=True, range=(0,1)))

        mixed_dls = get_mixed_dls( ts_dls, tab_dls, bs=128)
        if self.v:
            mixed_dls.show_batch()

        self.ts_dls = ts_dls
        self.tab_dls = tab_dls
        self.dls = mixed_dls

    def get_test_dls(self):
        """ test dataloaders using test splits
        """
        # create tab dataloader
        procs = [Categorify, FillMissing, Normalize]

        test_tab_dls = get_tabular_dls(self.tdf
                                        , procs=procs
                                        , cat_names=self.cat_names.copy()
                                        , cont_names=self.cont_names.copy()
                                        , y_names= self.target
                                        , splits= self.test_splits
                                    )
        
                # ts dataloader
        tfms  = [None, [Categorize()]]
        #test_ts_dls = get_ts_dls(self.X, self.y, splits=self.test_splits, tfms=tfms, batch_tfms=TSStandardize(by_var=True))
        test_ts_dls = get_ts_dls(self.X, self.y, splits=self.test_splits, tfms=tfms, batch_tfms=TSNormalize(by_var=True, range=(0,1)))
        # mix
        self.test_mixed_dls = get_mixed_dls( test_ts_dls, test_tab_dls)