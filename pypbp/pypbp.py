"""Main module."""
import os
import pandas as pd

location = os.path.dirname(os.path.realpath(__file__))
my_file = os.path.join(location, 'data', 'timeseries.xlsx')
data = pd.read_excel(my_file)
data = data.dropna()
real_dynamic = data['Sum - Taf']
global time_series
time_series = []

for i in range(len(real_dynamic.values)):
    
    time_series.append(float(real_dynamic.values[i]))