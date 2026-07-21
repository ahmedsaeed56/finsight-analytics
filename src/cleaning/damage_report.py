import pandas as pd 
import seaborn as sns 
from src.config import CUSTOMERS_RAW


raw=pd.read_csv(CUSTOMERS_RAW,skipinitialspace=True)
cust=raw.copy() 
print(cust.head(5)) 
