import kagglehub
import pandas as pd 
import numpy as np
import sklearn
import matplotlib
import seaborn

# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

churn = pd.read_csv(path)