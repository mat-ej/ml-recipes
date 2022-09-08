import pandas as pd
from fasttrees.fasttrees import FastFrugalTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from trees.features import *

'''
note - fftrees require python 3.5.6
'''

data = pd.read_csv('trees/data/mma.csv', parse_dates = ['DATE'])

target = data['WINNER']

features = data[red + blue]
fc = FastFrugalTreeClassifier(max_levels=10)

fc.fit(features, target.astype(bool).values)

fc.get_tree()