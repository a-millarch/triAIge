import pandas as pd
import numpy as np
from IPython.display import display

from src.data.creators import create_subset_mapping, process_patients_info, process_ppj

from src.data.tools import MultiLabelBinarizer, transposed_df, show_missing_per_col, find_columns_with_word
from src.data.list_dumps import pt_1, pt_2, pt_3
from src.features.loader import ABCD, sks_labels

import logging
from pathlib import Path
p = Path(__file__).parents[2]

# create logger object instead
logging.basicConfig(level = logging.INFO, 
                    filename=p.joinpath('logging/app.log'), 
                    filemode='w', format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

imported_cat_names = ABCD

class ppjDataset():
    """
    Base class for PPJ-dataset based on a population defined 
    """
    def __init__(self,default_mode=True, max_na:float = 0.25,
                  cat_names:list =imported_cat_names, limit_cats: bool = True, sks_labels = sks_labels):
        
        self.max_na = max_na
        self.subset = {}
        self.event_descriptions, self.subset_dict = create_subset_mapping()
        self.sks_labels = sks_labels

        if limit_cats:
            self.cat_names = cat_names
        else:
            self.cat_names = []

        self.cont_names = []

        if default_mode:
            self.collect_base_datasets()
            logging.info("Collected base datasets")
            self.collect_subsets()
            logging.info("Sorted PPJ into subsets")
            self.sort_subsets()
        else:
            pass
# dataset functions
    def __set_base_datasets__(self, ppj:pd.DataFrame, patients_info: pd.DataFrame):
        self.ppj = ppj
        self.patients_info = patients_info
    
    def collect_base_datasets(self, single_trajectory=False):
        """
        Call this to construct base datasets (patients info and ppj) if files are missing or need update.
        """
        patients_info = process_patients_info()
        ppj = process_ppj()

# ! probably be obsolete
        if single_trajectory:
            # some JournalIDs has more than one patient related, by mistake. Remove these. 
            _  = pd.DataFrame(patients_info.groupby(["JournalID","CPR"]).cumcount()+1, columns = ["JID_n"])
            patients_info = patients_info.join(_)
            drop_jids = patients_info[patients_info.JID_n >1]["JournalID"].unique()
            patients_info.drop(columns="JID_n", inplace=True)
            patients_info = patients_info[~patients_info.JournalID.isin(drop_jids)]
               
        self.__set_base_datasets__(ppj, patients_info)

    def collect_subsets(self):
        """
        Create, proces and collect subset from PPJ data
        """
        population_size = len(self.patients_info["JournalID"].unique())
        max_na = self.max_na
        self.na_dict = {}

        for name, code in self.subset_dict.items():
            # filter ppj by code
            if isinstance(code, list):
                self.subset[name] = self.ppj[self.ppj.EventCodeName.isin(code)].copy(deep=True)
            elif isinstance(code, str):
                self.subset[name] = self.ppj[self.ppj.EventCodeName == code].copy(deep=True)
            else: 
                logging.info("invalid code")
# refactor 
            df = self.subset[name]
            # remove if empty in dataset 
            
            if self.subset[name].empty:
               # print(f"Removed: {name: <50}- empty")
                del self.subset[name]
            else:
                # sort subset by time    
                self.subset[name].sort_values(["JournalID", "CreationTime"], inplace=True)  
                self.subset[name] = self.subset[name].merge(self.patients_info[["JournalID", "PID"]], on="JournalID", how="left")
                na_percentage = (1-(len(self.subset[name]["PID"].unique())  / population_size)) 
                self.na_dict[name] = na_percentage

                # remove if below threshold of max na
                if na_percentage > max_na:
                    #logging.info(f"Removed: {name: <50}- below threshold with missing percentage: ({na_percentage*100:.0f}%)")
                    del self.subset[name]
                    continue
                # remove unnecesary value columns and rename
                for column in self.subset[name].columns:
                    if all(self.subset[name][column].isnull()):
                        self.subset[name].drop(columns = column, inplace=True)
                    # if value column    
                    elif column.startswith("Value"):      
                        self.subset[name].rename(columns={column:"Value"}, inplace=True)
    
    def sort_subsets(self):
        ed = self.event_descriptions[self.event_descriptions.Tekst.isin(self.subset.keys())]
        self.subset_info_by_dtype = {}
        d = self.subset_info_by_dtype

        d["bools"] =  ed[ed["Datatype"] == "bool"]
        d["categoricals"] = ed[ed["Datatype"] == "listvalue"]
        d["numericals"] = ed[ed["Datatype"].isin(["float", "integer", "double"])]
        d["seq_numericals"] = ed[ed["Datatype"] == "seq_float"]
        d["datetime"] =  ed[ed["Datatype"] == "datetime"]

        d["composite"] =  ed[ed["Datatype"] == "composite"]

        d["string"] = ed[(ed["Datatype"] == "string")] #Leave this one out for now
        
        for type in d.keys():
            setattr(self, f"subset_{type}", {key: value for key,value in self.subset.items() if key in d[type]["Tekst"].tolist()})

        if len(self.cat_names) == 0:
            self.cat_names = list(self.subset_categoricals.keys()) # type: ignore 

# new class here? df-maker! children: tab and ts
# outcomes
    def add_outcome(self, outcome_mode:str = "categorical_procedure"):
        
        if outcome_mode == "TRANSFUSION":
            self.patients_info.loc[self.patients_info.JournalID.isin(pt_1), "TRANSFUSION_1H"] = 1
            self.patients_info["TRANSFUSION_1H"].fillna(0, inplace=True)

            self.patients_info.loc[self.patients_info.JournalID.isin(pt_2), "TRANSFUSION_2H"] = 1
            self.patients_info["TRANSFUSION_2H"].fillna(0, inplace=True)

            self.patients_info.loc[self.patients_info.JournalID.isin(pt_3), "TRANSFUSION_3H"] = 1
            self.patients_info["TRANSFUSION_3H"].fillna(0, inplace=True)        
        
        elif outcome_mode == "any_procedure":
            self.patients_info.loc[self.patients_info["ProcedureCode_1"].notnull(), ["label"]] = 1
            self.patients_info["label"].fillna(0,inplace=True)

        elif outcome_mode == "categorical_procedure":
            # move and import
            proc_cols = find_columns_with_word(self.patients_info, "ProcedureCode")
            
            for name, code in self.sks_labels.items():
                # Define a list of column names to search within
                columns_to_search = proc_cols

                # Define a function to check if any element in the specified columns starts with the search string
                def check_row(row):
                    for col in columns_to_search:
                        if str(row[col]).startswith(code):
                            return 1
                    return 0
                # Apply the check_row function to each row and store the result in a new column
                self.patients_info[name] = self.patients_info.apply(lambda row: check_row(row), axis=1)
        else: logging.error("invalid label")

# categorical general functions
    def rename_categorical_values(self, cat_names:list = [], as_category: bool = True):
        """
        Use event descriptions to rename values from float/code to tekst in a subset.
        """
        if not cat_names:
            cat_names = self.cat_names

        ed = self.event_descriptions
        ed = ed[ed.Værdier.notnull()][["Kode", "Tekst", "Værdier"]]

        values_to_text = {}

        ed = ed[ed.Tekst.isin(cat_names)]

        for name in ed["Tekst"]:
            ed_df = ed[ed["Tekst"] == name]
            key_value_pairs = ed_df["Værdier"].str.split(';')
            
            values_to_text[name]= {}
            for pair in key_value_pairs:

                for kv in pair:
                    keyval= kv.split("=")
                    key = float(keyval[0])
                    val =keyval[1]
                    values_to_text[name][key] = val
                    #values_to_text

        for f in cat_names:
            #ds.subset[f] = ds.subset[f][["JournalID", "Value"]]
            self.subset_categoricals[f]["Value"] = self.subset_categoricals[f]["Value"].replace(values_to_text[f]) # type: ignore
            if as_category:
                cats = self.subset_categoricals[f]["Value"].unique() # type: ignore
                self.subset_categoricals[f]["Value"] = pd.Categorical( self.subset_categoricals[f]["Value"], categories= cats, ordered = True)  # type: ignore
   
    def proces_categoricals(self, rename_values = True):
        """ Fixes temporal features, sorts values by severity and keeps highest. 
        Renames feature values from coded to descriptive (mainly useful for NLP strategies)"""
        for df in self.subset_categoricals.keys(): # type: ignore

            ss = self.subset_categoricals[df] # type: ignore
            # Overwrite CreationTime with ManualTime if available
            try: 
                ss.ManualTime.fillna(ss.CreationTime, inplace=True)
                ss["CreationTime"] = ss ["ManualTime"]
                del ss["ManualTime"]
            except:
                logging.warning("No column manualtime, proceed")
                pass

            # Keep highest severity value, drop duplicates
            self.subset_categoricals[df] = ss.sort_values(by=["JournalID", "Value"], ascending=True).drop_duplicates(subset="JournalID",keep='last') # type: ignore
        
        if rename_values:
            self.rename_categorical_values()

    def merge_categoricals(self):
        """ Merges categorical values onto main df. Use proces func beforehand if wanted"""
        for df in self.subset_categoricals.keys(): # type: ignore
            ss = self.subset_categoricals[df]  # type: ignore
            ss[df+"_Value"] = ss["Value"]
            del ss["Value"], ss["CreationTime"], ss["EventCodeName"]
            self.cat_names.remove(df)
            self.cat_names.append(df+"_Value")
            self.df = self.df.merge(ss.drop(columns=["PID", "Unnamed: 0"]), on= ["JournalID", "JournalID"], how="left")
            
class TabularDataset(ppjDataset):
    def __init__(self, max_na = 0.25 ,parent_default_mode = True ,default_mode=True):
        super().__init__(max_na=max_na, default_mode= parent_default_mode)

        self.df = self.patients_info.copy(deep=True)

        if not self.cat_names:
            self.cat_names= []

        if not self.cont_names:
            self.cont_names= []
        
        if default_mode:
            self.binarize_categoricals()
            self.numeric_sequence_to_tabular()
            #self.add_transfusion()
        else:
            pass

    def numeric_sequence_to_tabular(self):
        num_dfs = self.subset_seq_numericals # type: ignore
        for name in num_dfs:
            #print(name, "attempting description")
            
            tmp = num_dfs[name][["JournalID", "Value"]].groupby("JournalID").Value.describe().unstack(1).reset_index().pivot(index='JournalID', values=0, columns='level_0')
            
            tmp.reset_index(inplace=True)

            renamer = {"25%":f"{name}_25%",
                       "50%":f"{name}_50%",
                       "75%":f"{name}_75%",
                       "count":f"{name}_count",
                       "max":f"{name}_max",
                       "mean":f"{name}_mean",
                       "min":f"{name}_min",
                       "std":f"{name}_std"}
            tmp.rename(columns = renamer, inplace=True)
            self.cont_names.append(tmp.drop(columns="JournalID").columns.tolist())

            self.df = self.df.merge(tmp, on="JournalID", how="left")

    def binarize_categoricals(self):
        for df_name in self.subset_categoricals.keys():# type: ignore
# !!!
            if df_name == "Behandling før ankomst":
                continue
            #print(f"ohe for {df_name}")
            sdf = self.subset_categoricals[df_name] # type: ignore

            try:
                ohe_df = pd.get_dummies(sdf[["PID", "Value"]], columns=['Value'], prefix= df_name).groupby(['PID'], as_index=False).sum()
            except:
                continue

            self.cat_names.append(ohe_df.drop(columns="PID").columns.tolist())

            sdf.drop(columns=["Value","EventCodeName","CreationTime"], inplace=True)# type: ignore
            try:
                sdf.drop(columns=["ManualTime"], inplace=True)# type: ignore
            except:
                pass
            self.df = self.df.merge(ohe_df, on =["PID"], how="left")# type: ignore
                     
    def export_to_csv(self, mode="df"):
        
        if mode =="df":
            try: 
                export_path = "data/processed/df.csv"
                self.df.to_csv(p.joinpath(export_path))
            except:
                pass
     
class TimeSeriesDataset(ppjDataset):
    """
    to do: create feature dict and target dict in parent class"""
    def __init__(self, target="laparo"):
        super().__init__()
        
        self.target =target

    def bin_sequential_data(self, bin_freq = "1Min", keep_creation_time = False):
        dfs = ["M_Puls",  'M_NInv Sys Blodtryk', 'M_NInv Dia Blodtryk', 'M_SpO2' ]
        id_col = "JournalID"
        datetime_col = "CreationTime"

        df_dict = {key: self.subset[key] for key in dfs} 
        # Initialize an empty dataframe to store the final result
        final_df = pd.DataFrame()

        # Iterate through each dataframe and perform the grouping and aggregation
        for df_name, df in df_dict.items():
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            # Group by ID and create 2-minute bins for DateTime
            df_grouped = df.groupby([id_col, pd.Grouper(key=datetime_col, freq=bin_freq)])['Value'].max().reset_index()
            
            # Rename the 'Value' column with a suffix based on the dataframe name
            df_grouped = df_grouped.rename(columns={'Value': f'Value_{df_name}'})
            
            # Merge the grouped dataframe with the final result based on ID and DateTime
            if final_df.empty:
                final_df = df_grouped
            else:
                final_df = pd.merge(final_df, df_grouped, on=[id_col, datetime_col], how='outer')

# Careful here, might fuck with IDS 
        # Fill NaN values forward within the series
        final_df = final_df.fillna(method="ffill")

        final_df['timestep'] = final_df.groupby('JournalID')['CreationTime'].transform(lambda x: (x - x.min()).dt.total_seconds() / 120)
        if keep_creation_time:
            pass
        else:
            final_df.drop(columns="CreationTime", inplace=True)

        self.max_seq_len = final_df.groupby("JournalID").count().sort_values(by="JournalID", ascending=False).timestep.max()
        self.df = final_df 

    def __construct_features__(self, outcome_mode, bin_freq):
        self.add_outcome(outcome_mode = outcome_mode) # adds outcome to patients df
        self.bin_sequential_data(bin_freq= bin_freq) # creates self.df by binning
     
        # Create SI and append target to df
        self.df[["Value_M_NInv Sys Blodtryk", "Value_M_NInv Dia Blodtryk"]] = self.df[["Value_M_NInv Sys Blodtryk", "Value_M_NInv Dia Blodtryk"]].fillna(method="backfill")
        #self.df["SI"] = (self.df["Value_M_Puls"] / self.df["Value_M_NInv Sys Blodtryk"]).round(2)

        self.df = self.df.merge(self.patients_info[["JournalID", self.target]], how='left', on = "JournalID")

    def __df_to_long_df(self):
        """ TO DO! eliminate manual column names/ move to cfg
        """
        df = self.df.copy(deep=True)

        df.rename(columns={#"JournalID": "sample", 
                        "Value_M_Puls": "puls", 
                        "Value_M_NInv Sys Blodtryk" :"sys", 
                        "Value_M_NInv Dia Blodtryk": "dia",
                        "Value_M_SpO2": "sat"
                        }, inplace=True)

        df['sample'] = df.groupby('JournalID').ngroup()

        self.patient_map = df[["sample", "JournalID"]].copy(deep=True)
        self.patient_map = self.patient_map.drop_duplicates().reset_index(drop=True)
        del df["JournalID"]
        self.y_df = df[["sample", self.target]].drop_duplicates()
        #self.y_df[self.y_df["sample"].duplicated(keep=False)]

        self.df_dict = transposed_df(df, ["puls","sys","dia", "sat"]) #, "SI"])

        long_df = pd.concat(self.df_dict.values(), ignore_index=True)
        self.long_df = long_df.sort_values(by=["sample", "feature"]).reset_index(drop=True)

    def construct_df(self, 
                outcome_mode = "categorical_procedure", bin_freq = "30S",
                df_format = "long", 
                
                verbose=False ):
        
        self.__construct_features__(outcome_mode, bin_freq)
        if df_format == "long":
            self.__df_to_long_df()
            self.mp = show_missing_per_col(self.df_dict["puls"].drop(columns=["sample", "feature"]))
            if verbose: 
                print("\n\ndefine cutoff (cutoff_col_idx) for missing values, using TimeSeriesDataset.cut(n_cols= cutoff_col_idx)")
                display(self.mp.head(30))

    def cut(self,cutoff_col_idx =30, remove_short_nulls=False, fillmode = "mean"):
        try:
            self.long_df = self.long_df.iloc[:,:cutoff_col_idx+3]
# fill na was here
            self.long_df = self.long_df.merge(self.y_df,how="left", on="sample")#
            self.long_df.reset_index(drop=True, inplace=True)
            if fillmode == "trailingzero":
                logging.info("filling long df with trailing zeros")
                self.long_df.fillna(0, inplace=True)
# !!!!!!!!!!
            elif fillmode == "mean":
                data_cols = self.long_df.columns[2:-1]
                row_means = self.long_df[data_cols].mean(axis=1)
                self.long_df.fillna(row_means, inplace=True)
            else: pass
            
            if remove_short_nulls:
                short_positive = len(self.long_df[(self.long_df[self.target]==1.) & (self.long_df[5]==0 )]["sample"].unique())
                logging.info(short_positive)
                drop_samples = self.long_df[(self.long_df[self.target]==0.) & (self.long_df[5]==0 )]["sample"].unique()
                drop_samples = drop_samples[:-short_positive * 2]
                self.long_df = self.long_df[~self.long_df["sample"].isin(drop_samples)]

        except: 
            self.construct_df(df_format = "long")
            try: self.cut()
            except:  logging.error("unable to cut long df after attempting construction")

    def compute(self,cutoff_col_idx= 30, remove_short_nulls =True,fillmode = "mean",**kwargs ):
        self.construct_df(**kwargs)
        self.cut(cutoff_col_idx, remove_short_nulls= remove_short_nulls, fillmode= fillmode)