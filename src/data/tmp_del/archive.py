
####### graveyard below

def build_pulse(ppj, patients_info):
    pulse = ppj[ppj.EventCodeName =="OMI00001"].copy(deep=True).merge(patients_info[["JournalID","CPR"]], how='left', on='JournalID')
    pulse= pulse.sort_values(["JournalID", "CreationTime"])

    pulse["TS"]= pulse.groupby("JournalID").cumcount()
    return pulse


def build_doa(ppj):
    # DEAD ON ARRIVAL 
 
    #1=Genoplivning indstillet;2=Fortsat hjertestop, genoplivning fortsættes;4=Følelig puls - spontant kredsløb genoprettet;8=Patient vågen + GCS større end 8
    doa = ppj[(ppj.EventCodeName == "HAC00034") & (ppj.ValueFloat < 3)]
    return doa

def build_defib(ppj, patients_info):
    # Defib
    defib_codes = ["BHC00002", "HAC00005"]
    defib = ppj[ppj.EventCodeName.isin(defib_codes)].merge(patients_info[["JournalID","CPR"]], how='left', on='JournalID')
    len(defib)
    return defib


from src.data.datasets import process_patients_info, process_ppj
def get_datasets():
    patients_info = process_patients_info()
    ppj = process_ppj()

    pulse = build_pulse(ppj, patients_info)

    doa = build_doa(ppj)
    defib = build_defib(ppj,patients_info)


    datasets ={"patients_info": patients_info,
               "ppj": ppj,
               "pulse":pulse,
               "doa": doa,
               "defib": defib}

    return datasets

# Function to return the constant value columns of a given DataFrame
def remove_constant_value_features(df):
    return [e for e in df.columns if df[e].nunique() == 1]

