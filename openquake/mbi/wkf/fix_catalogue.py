import numpy as np
import pandas as pd
from openquake.baselib import sap

def fix_catalogue(fname):
    """
    Clean the catalogue by fixing cases where months or days are not set
    properly
    """

    df = pd.read_csv(fname)
    print(df.head())

    condition = ((df.month < 1) | (df.month > 12))
    df.month = np.where(condition, 1, df.month)

    condition = ((df.day < 1) | (df.day > 31))
    df.day = np.where(condition, 1, df.day)

    df.to_csv(fname)


def main(fname):
    fix_catalogue(fname)


main.fname = "Name of the file containing the catalogue"

if __name__ == '__main__':
    sap.run(main)
