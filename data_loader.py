import pandas as pd

# Load the Glass Identification Data Set csv data using the Pandas library
filename = 'dataset/glass.cvs'
df = pd.read_csv(filename)

# number of samples
N = df.shape[0]
# number of attributes
M = len(df.columns)

classNames = sorted(set(df['type']))