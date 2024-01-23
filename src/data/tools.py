import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


from pathlib import Path
p = Path(__file__).parents[2]

def pickle_loader(filename):
    if "/" in filename:
        data_loc = "data" 
    else:
        data_loc= "data/interim/"
    file_path = p.joinpath(data_loc, filename)
    #print("Attempting to load file from:", file_path)
    df = pd.read_pickle(file_path)
    #print("File succesfully loaded.\n")
    return df

def find_columns_with_word(dataframe, word):
    matching_columns = [col for col in dataframe.columns if word in col]
    return matching_columns

def værdi_to_dict(ds):
    hr_val_dict= {}

    for item in ds.subset_info_by_dtype["composite"].iloc[0]["Værdier"].split(";"):
        key, val = item.split("=",1)
        hr_val_dict[key+".0"] =val.replace(" /","").replace(" ", "_")
    return hr_val_dict

def cpr_tjek(df,col_name):
    return(len(df[df[col_name].astype(str).str.len() <10]))

def cpr_fix(df,col_name):
    n = cpr_tjek(df,col_name)
    print(f"\nAntal CPR (len < 10) rettet:{n}")
    
    df[col_name] = df[col_name].astype(str)
    df[col_name] = np.where(df[col_name].str.len().isin([9,]), '0'+ df[col_name], df[col_name])
    return df

class PrefixLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, prefix=None):
        self.prefix = prefix
        self.binarizer = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.binarizer.fit(X)
        return self

    def transform(self, X):
        binarized = self.binarizer.transform(X)
        if self.prefix is not None:
            # Create new column names with the specified prefix
            num_classes = binarized.shape[1]
            new_columns = [f"{self.prefix}_{i}" for i in range(num_classes)]
            binarized = pd.DataFrame(binarized, columns=new_columns) # type: ignore
        return binarized
    
    
def transposed_df(df, cols):
    df_dict = {}
    cnt = 0
    for c in cols: 
        cnt= cnt+1
        df_tmp=df[["sample", c]]
        transposed = df_tmp.groupby('sample')[c].apply(lambda df: df.reset_index(drop=True)).unstack().reset_index()
        transposed.insert(loc=1, column = "feature", value = cnt)
        df_dict[c] = transposed
    return df_dict



def show_missing_per_col(df):
    percent_missing = (df.isnull().sum() * 100 / len(df)).round(0).astype(int)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                    'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', ascending=True,inplace=True)
    return missing_value_df


def cpr_to_birthdate(input_df, cpr_col_name, new_col_name = "Birthdate"):
    #add 1900 year
    input_df[new_col_name] =  pd.to_datetime(input_df[cpr_col_name].map(lambda x: x[:4] + "19" +x[4:-4]), format="%d%m%Y", errors="coerce")
   
    # change to 2000 if rules (https://cpr.dk/media/12066/personnummeret-i-cpr.pdf)

    rules2000= [[36, 4000, 5000]
               ,[36, 9000, 10000]
               ,[57, 5000, 9000]]

    for i in rules2000:

        # fix 2000 kids

        input_df.loc[(input_df[cpr_col_name].map(lambda x: x[4:6]).astype(int) <= i[0])

                     & (input_df[cpr_col_name].map(lambda x: x[6:]).astype(int).between(i[1], i[2], inclusive='left'))

                     , new_col_name] =pd.to_datetime(input_df[cpr_col_name].map(lambda x: x[:4] + "20" +x[4:-4]), format="%d%m%Y", errors="coerce")      

    return input_df