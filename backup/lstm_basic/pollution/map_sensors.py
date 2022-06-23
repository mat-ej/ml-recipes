#!/usr/bin/env python
# coding: utf-8

# In[34]:


import folium
import pandas as pd


# In[37]:


sensors = pd.read_csv("sensor_coordinates.csv")
sensors


# In[61]:


m = folium.Map(location=[31.5, -98.5], zoom_start=7)
for i, x in sensors.iterrows():
    if i != 5:
        folium.Marker([x['latitude'], x['longitude']]).add_to(m)

folium.Marker(
    [sensors.iloc[5]['latitude'], sensors.iloc[5]['longitude']], 
    icon=folium.Icon(color="red", icon="screenshot", prefix='glyphicon')
).add_to(m)
m

