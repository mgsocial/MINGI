import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy import nan as NA
import matplotlib.pyplot as plt
import re
import mglearn

import os
os.chdir("C:\\Users\\kitcoop\\Desktop\\temp\\python")


def pdr(x) :
    csv = x + "." + "csv"
    c = pd.read_csv(csv, engine="python", index_col=0)
    return(c)


