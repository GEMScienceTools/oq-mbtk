import numpy as np
import pandas as pd
from openquake.baselib import sap

def fix_catalogue(fname):

    df = pd.read_csv(fname)
    print(df.head())

    condition = ((df.month < 1) | (df.month > 12))
    df.month = np.where(condition, 1, df.month)

    condition = ((df.day < 1) | (df.day > 31))
    df.day = np.where(condition, 1, df.day)

    df.to_csv(fname)

if __name__ == '__main__':
    sap.run(fix_catalogue)
