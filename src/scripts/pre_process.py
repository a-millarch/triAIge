import pandas as pd
from pathlib import Path
from tqdm import tqdm   
import os
#os.chdir("..")
print("\n",os.getcwd() )


from  src.data.datasets  import ppjDataset

ds = ppjDataset()

print(ds.subset.keys())

