import pandas as pd
import numpy as np
from IPython.display import display
import copy 

from src.data.creators import create_subset_mapping, process_patients_info, process_ppj
from src.data.tools import transposed_df, show_missing_per_col, find_columns_with_word
from src.data.list_dumps import pt_1, pt_2, pt_3
from src.features.loader import ABCD, sks_labels

from pathlib import Path
p = Path(__file__).parents[2]

import logging
from src.common.log_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

imported_cat_names = copy.deepcopy(ABCD)

class ppjDataset():
    """
    Base class for PPJ-dataset based on a population defined 
    """
    def __init__(self,
                 default_mode=True, max_na:float = 0.25,
                  cat_names:list =imported_cat_names, limit_cats: bool = True, sks_labels = sks_labels):
        
        self.max_na = max_na
        self.subset = {}
        self.event_descriptions, self.subset_dict = create_subset_mapping()
        self.sks_labels = sks_labels

        if limit_cats:
            self.limit_cats = True
            self.cat_names = copy.deepcopy(cat_names)
        else:
            self.limit_cats = False
            self.cat_names = []

        self.cont_names = []

        if default_mode:
            self.collect_base_datasets()
            logger.info("Collected base datasets")
            self.collect_subsets()
            self.clean_sequentials()
            logger.info("Collected, cleaned and sorted PPJ into subsets")
            self.sort_subsets()
        
        else:
            pass
# dataset functions
    def __set_base_datasets__(self, ppj:pd.DataFrame, patients_info: pd.DataFrame):
        self.ppj = ppj
        self.patients_info = patients_info
    
    def collect_base_datasets(self, 
                              patients_info_file_name ="patients_info", 
                              ppj_file_name ="ppj",
                              single_trajectory=False):
        """
        Call this to construct base datasets (patients info and ppj) if files are missing or need update.
        """
        patients_info = process_patients_info(file_name = patients_info_file_name)
        ppj = process_ppj(file_name = ppj_file_name)

# ! probably be obsolete
        if single_trajectory:
            # some JournalIDs has more than one patient related, by mistake. Remove these. 
            _  = pd.DataFrame(patients_info.groupby(["JournalID","PID"]).cumcount()+1, columns = ["JID_n"])
            patients_info = patients_info.join(_)
            drop_jids = patients_info[patients_info.JID_n >1]["JournalID"].unique()
            patients_info.drop(columns="JID_n", inplace=True)
            patients_info = patients_info[~patients_info.JournalID.isin(drop_jids)]
               
        self.__set_base_datasets__(ppj, patients_info)

    def collect_subsets(self):
        """
        Create, proces and collect subset from PPJ data
        """
        logger.debug("Intializing collect_subsets()")
        
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
                logger.error(f"invalid code {code}")
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
                    #logger.info(f"Removed: {name: <50}- below threshold with missing percentage: ({na_percentage*100:.0f}%)")
                    del self.subset[name]
                    continue
                # remove unnecesary value columns and rename
                for column in self.subset[name].columns:
                    if all(self.subset[name][column].isnull()):
                        self.subset[name].drop(columns = column, inplace=True)
                    # if value column    
                    elif column.startswith("Value"):      
                        self.subset[name].rename(columns={column:"Value"}, inplace=True)
            logger.debug(f"Collected: {name}")
        logger.info("Collected subsets")
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
        else: pass
        
        if self.limit_cats:
            # reduce categoricals to predefined list no matter what if limit-categoricals set true
            r = {key: value for key, value in self.subset_categoricals.items() if key in self.cat_names} # type: ignore
            self.subset_categoricals = r
        logger.info("Sorted subsets")

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
        else: logger.error(f"invalid outcome cretion mode: {outcome_mode}")
        logger.info("Added outcomes")
# cleaners
    def clean_sequentials(self):
        """ Removes outliers/noise for selected sequential input features"""
        self.subset["M_Puls"] =self.subset["M_Puls"][ self.subset["M_Puls"].Value.between(0.,220.)]
        self.subset["M_SpO2"] = self.subset["M_SpO2"][self.subset["M_SpO2"].Value.between(0.,100.)]
        self.subset["M_NInv Sys Blodtryk"] = self.subset["M_NInv Sys Blodtryk"][self.subset["M_NInv Sys Blodtryk"].Value.between(0.,250.)]
        self.subset["M_NInv Dia Blodtryk"] = self.subset["M_NInv Dia Blodtryk"][self.subset["M_NInv Dia Blodtryk"].Value.between(0.,160.)]
        logger.info("Cleaned sequential subsets for outliers")
class TabularDataset():
    def __init__(self, base: ppjDataset | None = None, max_na = 0.25 ,parent_default_mode = True ,default_mode=True):

        if isinstance(base, ppjDataset):
            self.base = base
        else: 
            logger.info("No ppjDataset object passed, creating from default values.")
            self.base = ppjDataset()

        self.df = self.base.patients_info.copy(deep=True)

        if not self.base.cat_names:
            self.base.cat_names= []

        if not self.base.cont_names:
            self.base.cont_names= []
        
        if default_mode:
            self.binarize_categoricals()
            self.numeric_sequence_to_tabular()
            #self.add_transfusion()
        else:
            pass

    def numeric_sequence_to_tabular(self):
        num_dfs = self.base.subset_seq_numericals # type: ignore
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
            self.base.cont_names.append(tmp.drop(columns="JournalID").columns.tolist())

            self.df = self.df.merge(tmp, on="JournalID", how="left")

# categorical general functions
    def rename_categorical_values(self, cat_names:list = [], as_category: bool = True):
        """
        Use event descriptions to rename values from float/code to tekst in a subset.
        """
        if not cat_names:
            cat_names = self.base.cat_names

        ed = self.base.event_descriptions
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
            self.base.subset_categoricals[f]["Value"] = self.base.subset_categoricals[f]["Value"].replace(values_to_text[f]) # type: ignore
            if as_category:
                cats = self.base.subset_categoricals[f]["Value"].unique() # type: ignore
                self.base.subset_categoricals[f]["Value"] = pd.Categorical( self.base.subset_categoricals[f]["Value"], categories= cats, ordered = True)  # type: ignore
        logger.info("Renamed categorical values to natural language")

    def proces_categoricals(self, rename_values = True):
        """ Fixes temporal features, sorts values by severity and keeps highest. 
        Renames feature values from coded to descriptive (mainly useful for NLP strategies)"""
        for df in self.base.subset_categoricals.keys(): # type: ignore

            ss = self.base.subset_categoricals[df] # type: ignore
            # Overwrite CreationTime with ManualTime if available
            try: 
                ss.ManualTime.fillna(ss.CreationTime, inplace=True)
                ss["CreationTime"] = ss ["ManualTime"]
                del ss["ManualTime"]
            except:
                logger.warning(f"No column manualtime for {df}, proceed")
                pass

            # Keep highest severity value, drop duplicates
            self.base.subset_categoricals[df] = ss.sort_values(by=["JournalID", "Value"], ascending=True).drop_duplicates(subset="JournalID",keep='last') # type: ignore
        
        if rename_values:
            self.rename_categorical_values()

    def merge_categoricals(self):
        """ Merges categorical values onto main df. Use proces func beforehand if wanted"""
        for df in self.base.subset_categoricals.keys(): # type: ignore
            ss = self.base.subset_categoricals[df]  # type: ignore
            ss[df+"_Value"] = ss["Value"]
            del ss["Value"], ss["CreationTime"], ss["EventCodeName"]
            self.base.cat_names.remove(df)
            self.base.cat_names.append(df+"_Value")
            self.df = self.df.merge(ss.drop(columns=["PID", "Unnamed: 0"]), on= ["JournalID", "JournalID"], how="left")

    def binarize_categoricals(self):
        for df_name in self.base.subset_categoricals.keys():# type: ignore
# !!!
            if df_name == "Behandling før ankomst":
                continue
            #print(f"ohe for {df_name}")
            sdf = self.base.subset_categoricals[df_name] # type: ignore

            try:
                ohe_df = pd.get_dummies(sdf[["PID", "Value"]], columns=['Value'], prefix= df_name).groupby(['PID'], as_index=False).sum()
            except Exception as e:
                logger.exception("An error occurred: %s", str(e))
                continue

            self.base.cat_names.append(ohe_df.drop(columns="PID").columns.tolist())

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
class TimeSeriesDataset():
    """
    to do: create feature dict and target dict in parent class"""
    def __init__(self, base: ppjDataset | None, target="any_major", cfg = None):
      
        if isinstance(base, ppjDataset):
            self.base = base
        else: 
            logger.info("No ppjDataset object passed, creating from default values.")
            self.base = ppjDataset()

        self.target =target
        if isinstance(target, str):
            self.multilabel = False
        else:
            self.multilabel = True

        self.patients_info = self.base.patients_info.copy(deep=True)
        self.cfg = cfg

    def specify_sequential_features(self):
        self.original_seq_df_names = ["M_Puls",  'M_NInv Sys Blodtryk', 'M_NInv Dia Blodtryk', 'M_SpO2' ]
        self.df_dict = {key: self.base.subset[key] for key in self.original_seq_df_names} 

    def get_long_sequence_ids(self, limit = 100):
        drop_ids = []
        for df_name, df in self.df_dict.items():
            # reduce for performance, first ensure not duplicates then drop lenghty observations
            self.df_dict[df_name] = df.drop_duplicates()
            obs_count = pd.DataFrame(df.groupby("JournalID")["Value"].count()).reset_index()
            drop_ids.append(obs_count[obs_count["Value"] > limit]["JournalID"].unique())
        # Concatenate the arrays into a single array
        concatenated_array = np.concatenate(drop_ids)
        # Get unique values
        unique_values = np.unique(concatenated_array)
        # Convert the unique values back to a regular Python list if needed
        self.long_sequence_ids = list(unique_values)
        logger.info(f"Identified {len(self.long_sequence_ids)} JournalIDs with sequential data above limit: {limit}")

    def reduce_population_by_sequential_length(self, upper_seq_limit=100):
        
        self.get_long_sequence_ids(limit = upper_seq_limit)
        for df_name, df in self.df_dict.items():
            # drop if in long sequence
            self.df_dict[df_name]  = df[~df.JournalID.isin(self.long_sequence_ids)]
        logger.info(f"Removed {len(self.long_sequence_ids)} journalIDs due to sequences length above limit")


    def bin_sequential_data(self, bin_freq = "1Min", keep_creation_time = False):
        """ 
        """
        logger.info(f"Initialing sequential binning using {bin_freq} bins")
       
        id_col = "JournalID"
        datetime_col = "CreationTime"
       
        # Initialize an empty dataframe to store the final result
        final_df = pd.DataFrame()

        # Iterate through each dataframe and perform the grouping and aggregation
        for df_name, df in self.df_dict.items():
            
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            # Group by ID and create 2-minute bins for DateTime

            logger.debug(f"binning {df_name}")
            df_grouped = df.groupby([id_col, pd.Grouper(key=datetime_col, freq=bin_freq)])['Value'].max().reset_index()
            
            # Rename the 'Value' column with a suffix based on the dataframe name
            df_grouped = df_grouped.rename(columns={'Value': f'Value_{df_name}'})
            
            # Merge the grouped dataframe with the final result based on ID and DateTime
            if final_df.empty:
                final_df = df_grouped
            else:
                final_df = pd.merge(final_df, df_grouped, on=[id_col, datetime_col], how='outer')
                logger.debug(f"merged {df_name} onto ldf")
        logger.debug(f"Binning finalized.")

# Careful here, might fuck with IDS 
        # Fill NaN values forward within the series
        final_df = final_df.fillna(method="ffill")

        logger.debug("creating timesteps")
        final_df['timestep'] = final_df.groupby(['JournalID'])['CreationTime'].rank(method='dense') \
                                .sub(1).astype(int)
        #final_df['timestep'] = final_df.groupby('JournalID')['CreationTime'].transform(lambda x: (x - x.min()).dt.total_seconds() / 120)
        if keep_creation_time:
            pass
        else:
            final_df.drop(columns="CreationTime", inplace=True)
        logger.debug("setting max sequence length to object (self.max_seq_len)")    
        self.max_seq_len = final_df.groupby("JournalID").count().sort_values(by="JournalID", ascending=False).timestep.max()
        self.df = final_df 
        logger.debug("self.df set")

        if self.multilabel:
            self.df = self.df.merge(self.patients_info[["JournalID",]+ self.target], how='left', on = "JournalID") #type: ignore
        else:
            self.df = self.df.merge(self.patients_info[["JournalID", self.target]], how='left', on = "JournalID")
  #  def construct_features(self, bin_freq):
       
       # self.bin_sequential_data(bin_freq= bin_freq) # creates self.df by binning
     
        #self.df[["Value_M_NInv Sys Blodtryk", "Value_M_NInv Dia Blodtryk"]] = self.df[["Value_M_NInv Sys Blodtryk", "Value_M_NInv Dia Blodtryk"]].fillna(method="backfill")
        #self.df["SI"] = (self.df["Value_M_Puls"] / self.df["Value_M_NInv Sys Blodtryk"]).round(2)
    def df_to_long_df(self):
        """ TO DO! eliminate manual column names/ move to cfg
        """
        logger.info(f"creating long df")
        df = self.df.copy(deep=True)

# MOVE THIS DAMMIT
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
        if self.multilabel:
            self.y_df = df[["sample",] +self.target].drop_duplicates() #type: ignore
        else:
            self.y_df = df[["sample", self.target]].drop_duplicates()
        #self.y_df[self.y_df["sample"].duplicated(keep=False)]

        self.df_dict = transposed_df(df, ["puls","sys","dia", "sat"]) #, "SI"])

        long_df = pd.concat(self.df_dict.values(), ignore_index=True)
        self.long_df = long_df.sort_values(by=["sample", "feature"]).reset_index(drop=True)

    def construct_df(self, 
                     bin_freq = "30S",
                     verbose=False ):
        
        self.bin_sequential_data(bin_freq)
        self.df_to_long_df()
        self.mp = show_missing_per_col(self.df_dict["puls"].drop(columns=["sample", "feature"]))
        #logger.info("\n\ndefine cutoff (cutoff_col_idx) for missing values, using TimeSeriesDataset.cut(n_cols= cutoff_col_idx)")
            #display(self.mp.head(30))

# fill trail missing
    def trailing_fill(self, fillmode = "zero"):
            if fillmode == "zero":
                logger.info("padding with zeroes")
                self.long_df.fillna(0, inplace=True)

            elif fillmode == "ffill":
                logger.info("padding with forward fill")
                self.long_df.fillna(method= 'ffill', inplace=True)

            elif fillmode == "nans":
                logger.info("padding with float(nan)s")
                self.long_df.fillna(value=float('nan'), inplace=True)
            
            elif fillmode =="pass":
                logger.info("not filling trailing missings")
                pass

            else: logger.error("unable to trail fill sequentials")

    def remove_short_sequences(self, limit=5):
        """ Remove any samples where sequence is shorter than limit
        to do: from cfg define by minutes"""
        
        drop_ids = self.long_df[(self.long_df[limit] == 0) | (self.long_df[limit].isnull()) ]["sample"].unique()
        self.long_df = self.long_df[~self.long_df["sample"].isin(drop_ids)].reset_index(drop=True)
        logger.info(f"Dropped {len(drop_ids)} entries due to short sequences.")

# fix later    
    def __ldf_samples_to_undersample__(self,frac):
        """ assumes "_major" in target... """
        if self.multilabel:
            df = self.long_df[self.target + ["sample"]].drop_duplicates() # type: ignore
            # Determine if any of the 'target' columns is equal to 1
            mask = (df.filter(like='_major') == 0).all(axis=1)

            # Filter the DataFrame based on the mask
            result = df[mask]
            random_sample = result.sample(frac=frac, random_state=42)

            # Extract the 'sample' column from the result
            drop_samples = random_sample['sample'].unique()#.tolist()

            return drop_samples
        else: logger.error("attempted to use _ldf_samples_to_undersample function on str target")

    def cut(self,cutoff_col_idx =30, undersampling=None):
        try:
            self.long_df = self.long_df.iloc[:,:cutoff_col_idx+3]

            # some samples have duplicates where target is both 0 and 1. Keep where 1
            if self.multilabel:
                pass #probably still needs handling though
            else:
                self.y_df = self.y_df.sort_values(by=["sample", self.target]).drop_duplicates(subset="sample", keep="last")
            
            # now merge target onto the long df
            self.long_df = self.long_df.merge(self.y_df,how="left", on="sample")#
            self.long_df.reset_index(drop=True, inplace=True)
 
            if undersampling == "short":
                short_positive = len(self.long_df[(self.long_df[self.target]==1.) & (self.long_df[5]==0 )]["sample"].unique())
                logger.info(f"Undersampled negative class by N: {short_positive}")
                drop_samples = self.long_df[(self.long_df[self.target]==0.) & (self.long_df[5]==0 )]["sample"].unique()
                drop_samples = drop_samples[:-short_positive * 2]
                self.long_df = self.long_df[~self.long_df["sample"].isin(drop_samples)]

            elif isinstance(undersampling, float) and undersampling >0.0:
            
                if self.multilabel:
                    drop_samples = self.__ldf_samples_to_undersample__(frac = undersampling)
                else :
                    drop_samples = self.long_df[(self.long_df[self.target]<1)].sample(frac=undersampling, random_state=42)["sample"].unique()
         
                self.long_df = self.long_df[~self.long_df["sample"].isin(drop_samples)]#type:ignore
                logger.info(f"Undersampled negative class by N={len(drop_samples)} | frac={undersampling}")#type:ignore

            else: pass

        except: 
            self.construct_df()
            try: self.cut()
            except:  logger.error("unable to cut long df after attempting construction")



    def compute(self,cutoff_col_idx= 30, 
                undersampling =None, 
                fillmode = "trailingzero", 
                upper_seq_limit=100,
                **kwargs ):
        self.specify_sequential_features()
        self.reduce_population_by_sequential_length(upper_seq_limit=upper_seq_limit)

        self.construct_df(**kwargs) # bins, transform to long, creates missing overview (self.mp)

        self.trailing_fill(fillmode= fillmode)
        self.remove_short_sequences(limit = 5)
        self.cut(cutoff_col_idx, undersampling= undersampling)