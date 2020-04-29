import pandas as pd
from os.path import join
import misc.constants as cs

def load_idiab(dataset, subject):
    """
    Load a IDIAB file into a dataframe
    :param dataset: name of dataset
    :param subject: name of subject
    :param day_len: length of day scaled to sampling frequency
    :return: dataframe
    """
    df = pd.read_csv(join(cs.path, "data", dataset, "IDIAB_" +subject + ".csv"), header=0)
    df = df.drop("index",axis=1)
    df.datetime = pd.to_datetime(df.datetime)
    return df