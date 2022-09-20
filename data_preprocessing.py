import pandas as pd

# Load the Glass Identification Data Set csv data using the Pandas library
filename = 'dataset/glass.data'

# Add header
df = pd.read_csv(filename, header=None, names=["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "type"])

# Remove id column
df.drop(columns=["id"], inplace=True)

# Remove data samples that are labeled as 
#   -- 4 vehicle_windows_non_float_processed (none in this database)
#   -- 5 containers
#   -- 6 tableware
#   -- 7 headlamps
# and keep data samples labeled as
#   -- 1 building_windows_float_processed
#   -- 2 building_windows_non_float_processed
#   -- 3 vehicle_windows_float_processed

df.drop(df[(df.type == 4) | (df.type == 5) | (df.type == 6) | (df.type == 7)].index, inplace=True)

# From the remaining three classes:
#   -- 1 building_windows_float_processed
#   -- 2 building_windows_non_float_processed
#   -- 3 vehicle_windows_float_processed
# combine classes 1 and 3 to have only two labels
#   -- 1 windows_float_processed
#   -- 2 windows_non_float_processed

df.type.replace({3:1}, inplace=True)

# Save data set as cvs file
df.to_csv("dataset/glass.cvs", index=False)
