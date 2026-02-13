
import os
import numpy as np
import pandas as pd


def merge_ses_event_rupture(event_file: str, rupture_file: str) -> pd.DataFrame:
    """
    Merging of earthquake ruptures and events of Event-based PSHA using 'rup_id' as key.

    Params:
        event_file : Path to SES event csv
        rupture_file : Path to SES rupture csv

    Returns:
        pd.DataFrame: Merged catalogue containing events matched with ruptures
    """
    events = pd.read_csv(event_file)
    ruptures = pd.read_csv(rupture_file)
    return events.merge(ruptures, on="rup_id", how="left")


def add_random_datetime(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assigning random date time (month/day/hour/minute/second)

    Params:
        df : pd.DataFrame (SES catalogue containing a 'year' column)
        seed : int (Random seed for reproducibility)

    Returns:
        pd.DataFrame: Dataframe with added datetime columns and formatted "Date"
    """
    np.random.seed(seed)
    n = len(df)

    months = np.random.randint(1, 13, size=n)
    days = np.zeros(n, dtype=int)

    for i in range(n):
        year = df.iloc[i]["year"]
        month = months[i]

        if month in (1, 3, 5, 7, 8, 10, 12):
            max_day = 31
        elif month in (4, 6, 9, 11):
            max_day = 30
        else:  # February
            max_day = 29 if ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)) else 28

        days[i] = np.random.randint(1, max_day + 1)

    df["month"] = months
    df["day"] = days
    df["hour"] = np.random.randint(0, 24, size=n)
    df["minute"] = np.random.randint(0, 60, size=n)
    df["second"] = np.random.randint(0, 60, size=n)

    df["Date"] = df.apply(lambda r: f"{int(r['month']):02d}/{int(r['day']):02d}/{int(r['year']):04d}", axis=1)
    return df


def convert_to_hmtk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converting SES merged catalogue to HMTK-compatible column names.

    Params:
        df : pd.DataFrame (Merged SES dataframe)

    Returns:
        pd.DataFrame (Reformatted dataframe following HMTK format)
    """
    df = df.rename(columns={
        "centroid_lon": "longitude",
        "centroid_lat": "latitude",
        "mag": "magnitude",
        "centroid_depth": "depth",
        "event_id": "eventID"
    })

    cols = [
        "longitude", "latitude",
        "year", "month", "day", "hour", "minute", "second",
        "magnitude", "depth",
        "source_id", "eventID", "rlz_id", "rup_id", "ses_id",
        "multiplicity", "trt", "strike", "dip", "rake",
        "Date"
    ]

    return df.reindex(columns=cols)


def build_hmtk_ses_catalogue(event_file: str,
                             rupture_file: str,
                             output_file: str,
                             seed: int = 42) -> str:
    """
    Building an HMTK-formatted SES catalogue from OpenQuake 
    SES event/rupture outputs.

    Params:
        event_file : Path to SES event file
        rupture_file : Path to SES rupture file
        output_file : Output CSV path
        seed : Random seed

    Returns:
        Path to the generated SES catalogue CSV
    """
    df = merge_ses_event_rupture(event_file, rupture_file)
    df = add_random_datetime(df, seed=seed)
    df = convert_to_hmtk(df)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df.to_csv(output_file, index=False)

    return output_file
