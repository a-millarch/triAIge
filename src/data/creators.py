import pandas as pd
import numpy as np

from pathlib import Path
p = Path(__file__).parents[2]

from src.data.tools import pickle_loader, cpr_fix
from src.data.mapping import month_dict

import logging
from src.common.log_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def construct_patients_info_df_sp(drop_CPR =True, drop_JID_duplicates =True, sp_file_name = 'coded'):
    """Construct patients info from a population defined by SP-data from cpr_df.pkl by raw PPJ data"""

    ppj = pickle_loader ("cpr_df.pkl")
    sp_path = p.joinpath("data/raw/"+sp_file_name+".csv")
    sp = pd.read_csv(sp_path, sep =";", encoding="UTF8")
    try: 
        sp.drop(columns="Unnamed: 0", inplace=True)
    except:
        pass

    # Clean and remove empty entries (when CPR is registrered later on)  
    ppj["CPR"] = ppj["ValueString"].str.replace('"', '') 
    ppj = ppj[ppj.ValueString != '""']
    ppj = cpr_fix(ppj, "CPR")

    # fixing datetimes 
    ppj.CreationTime.replace(month_dict, regex=True, inplace=True)
    ppj["CreationTime_dt"] = pd.to_datetime(ppj.CreationTime, format="%d%m%Y:%H:%M:%S.%f")

    # merge dataframes
    merged = sp.merge(ppj[["CPR", "JournalID"]], on ="JournalID", how="left")
    merged.drop_duplicates(inplace=True)
    patients_info = merged.drop(columns=["ProcedureCode", "ProcedureName", "bd", "ID", "delta_t"], errors="ignore")
    # ! not proper age but aprox
    patients_info["age"] = ((pd.to_datetime(patients_info.ServiceDate) - pd.to_datetime(patients_info.Fødselsdato)).dt.days/365).round(0)
    
    # create PID
    patients_info["PID"], _ = pd.factorize(patients_info["CPR"])
    
    if drop_CPR:
        del patients_info["CPR"]

    # some JournalIDs have mulitple CPR. Discard these as we cannot distinguish observations between patients
    if drop_JID_duplicates:
        drop_jids = patients_info[patients_info.JournalID.duplicated(keep=False)]["JournalID"]
        patients_info = patients_info[~patients_info.JournalID.isin(drop_jids)]
    return patients_info


def construct_ppj_collection(patients_info_df = None, 
                             ABSOLUTE_FILE_PATH = Path(__file__).parents[2].joinpath("data/raw/PreHospitalAI_utf8.csv"), 
                             chunksize = 1000):
    """
    ABSOLUTE_FILE_PATH: Filepath for raw, complete ppj .csv file.

    As in notebooks/extract.ipynb
    """
    if  isinstance(patients_info_df, pd.DataFrame):
        patients_info = patients_info_df
    else:
        patients_info = process_patients_info()

    journal_ids = patients_info["JournalID"].unique().tolist() # type: ignore

    iter_csv = pd.read_csv(ABSOLUTE_FILE_PATH, encoding="UTF8", delimiter=";", chunksize=chunksize)
    df = pd.concat([chunk[chunk["JournalID"].isin(journal_ids)] for chunk in iter_csv])

        # reformat datetime cols
    for dt in ["CreationTime", "ManualTime"]:
        df[dt].replace(month_dict, regex=True, inplace=True)
        df[dt] = pd.to_datetime(df[dt], format="%d%m%Y:%H:%M:%S.%f")

    # replace emptry strings with actual nan    
    df.replace('""', np.nan, inplace=True)
    # Fix composite values, temporary solution
    df = fix_composites(df)

    # Remove CPR from  PPJ-collection
    df = df[df.EventCodeName != "PAT00013"]
    
    return df

# !!! Constructor needs update for pretrain

def process_patients_info(file_name ="patients_info"):
    file_path = p.joinpath("data/interim/", file_name+".csv")
        # load precursor to patients_info df and create
    try:
        patients_info = pd.read_csv(file_path, encoding="utf-8", sep=";")
        logger.debug(f"Succesfully loaded patients info file from {file_path}")
    except:
        logger.warning(f'Could not find {file_path}\nConstructing patients info dataframe. This might take a while.')
        patients_info = construct_patients_info_df_sp()
        patients_info.to_csv(p.joinpath("data/interim/", file_name+".csv"), encoding="utf-8", sep=";")
       
    return patients_info

def process_ppj(file_name ="ppj"):
    file_path = p.joinpath("data/interim/", file_name+".csv")
    try : 
        ppj = pd.read_csv(p.joinpath(file_path), encoding="utf-8", sep=";")
        logger.debug(f"Succesfully loaded PPJ file from {file_path}")
    except:
        logger.warning(f'Could not find {file_path}\nConstructing PPJ dataframe. This might take a while.')
        ppj = construct_ppj_collection()
        ppj.to_csv(p.joinpath("data/interim/", file_name+".csv"),  encoding="utf-8", sep=";")
    return ppj


def fix_composites(df, mode ="collapse"):
    """ 
    Mode: either "expand" or "collapse"
    
    """    
    for code in ["OMI00008","OMI00009","OMI00010"]:
        val_strings = df[(df.EventCodeName == code)].ValueString.unique()
        
        # if only one value and its not nan, make it nan
        if len(val_strings)==1 and not df[(df.EventCodeName == code)].ValueString.isna().all():
            df.loc[df.EventCodeName == code, ["ValueString"]] = np.nan
            
        # if more than one value either expand into several colunms or collapse into one by setting ValueString to NaN.
        elif len(val_strings)>1:
            if mode == "expand":
                for val in val_strings:
                    df.loc[(df.EventCodeName == code) & (df.ValueString ==val), ["EventCodeName", "ValueString"]] = [code+"_"+val.replace('"',''), np.nan]
            elif mode == "collapse":
                df.loc[df.EventCodeName == code, ["ValueString"]] = np.nan
                    
    return df

def create_subset_mapping(modified=True, prioritize=True):
    """ Event description mapping for defining subsets based on data types """
    # Use the by-hand modified event description
    if modified:
        event_description_path = "data/data_dumps/event_descriptions_modified.xlsx"
        ed_pre = pd.read_excel(p.joinpath(event_description_path), sheet_name="Prædefinerede eventkoder", engine="openpyxl")
        if prioritize: 
            ed_pre = ed_pre[ed_pre["Priority"] == 1]
    else:
        event_description_path = "data/data_dumps/event_descriptions.xlsx"
        ed_pre = pd.read_excel(p.joinpath(event_description_path), sheet_name="Prædefinerede eventkoder", engine="openpyxl")
    # read both sheets and merge
    ed_vitals = pd.read_excel(p.joinpath(event_description_path), sheet_name="Eventkoder Vitaldata", engine="openpyxl")
    #merge
    ed = pd.concat([ed_pre, ed_vitals])
    keep_columns = ["Tekst", "Kode"]
    subset_mapping_dict = ed[keep_columns].set_index("Tekst").to_dict()
    return ed, subset_mapping_dict["Kode"]

# move this
def add_calendar_cols(df, dt_col_name):
    df["Year"] = df[dt_col_name].dt.year
    df["Month"] = df[dt_col_name].dt.month
    df["Day"] = df[dt_col_name].dt.day
