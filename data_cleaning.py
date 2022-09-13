import numpy as np
import pandas as pd

# number of observations saved for each severity category
NUMBER_SAVED = 1000

# Load the US Accidents csv data using the Pandas library
filename = 'dataset/US_Accidents_Dec21_updated.csv'
df = pd.read_csv(filename)
# print(df)
print(df.shape)

# We extract the attribute names:
attributeNames = np.asarray(df.columns)
print(attributeNames)

# Dropping unwanted columns
dropped_cols = [['ID'], ['Start_Time'], ['End_Time'], ['Start_Lat'], ['Start_Lng'], ['End_Lat'], ['End_Lng'],
                ['Description'], ['Number'], ['Street'], ['City'], ['County'], ['State'], ['Zipcode'], ['Country'],
                ['Timezone'], ['Airport_Code'], ['Weather_Timestamp'], ['Wind_Direction']]


for i in range(len(dropped_cols)):
    df = df.drop(dropped_cols[i][0], axis=1)

print(df.shape)

# here it seems precipitation has a lot of NaN values, maybe we have to drop it
# df = df[:200]
# print(df.to_string())
# Also our y will be the severity, which now is the first column

# Seeing what percentage of this column is comprised of NaN
print(df['Precipitation(in)'].isna().sum()/df.shape[0] * 100)
# the result is almost 20%, maybe we will drop it later

# This being such a huge data set, with almost 3 million observations, we would like to keep a very small subset,
# somewhere around 4000 observations at first, so it is easier to work with

# First we drop all the rows that have NaN in them
df2 = df.dropna().reset_index(drop=True)
# print(df2.to_string())
# We still have more than 2 million observations remaining
print(df2.shape)

# Then we keep 1000 observations out of each severity category
sev_df = df2[df2.groupby(['Severity']).cumcount() < NUMBER_SAVED]
# The next two lines are just to make sure we did it correctly
# print(sev_df.shape)
# sev_df = sev_df[:200]
# print(sev_df.to_string())

# Dropping columns that have only one unique value
print(sev_df['Roundabout'].unique())
sev_df = sev_df.drop('Roundabout', axis=1)
print(df['Turning_Loop'].unique())
sev_df = sev_df.drop('Turning_Loop', axis=1)

# Now we save the new data set
saved_filename = 'dataset/US_Accidents_small.csv'
sev_df.to_csv(saved_filename)
