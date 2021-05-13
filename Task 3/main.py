import numpy as np
import pandas as pd

# Read the data and remove the first Row
train_data = pd.read_csv("train.csv", delimiter = ',').values
test_data = pd.read_csv("test.csv", delimiter = ',').values

Y_train = [row[1] for row in train_data]

Alph_Map = {
    'A':0,
    'R':1,
    'N':2,
    'D':3,
    'C':4,
    'Q':5,
    'E':6,
    'G':7,
    'H':8,
    'I':9,
    'L':10,
    'K':11,
    'M':12,
    'F':13,
    'P':14,
    'S':15,
    'T':16,
    'W':17,
    'Y':18,
    'V':19
}

X_train = np.zeros((len(train_data), 4*len(Alph_Map)))
X_test = np.zeros((len(test_data), 4*len(Alph_Map)))


for i, line in enumerate(train_data):
    for j, c in enumerate(line[0]):
        X_train[i][j*len(Alph_Map)+Alph_Map[c]] = 1

for i, line in enumerate(test_data):
    for j, c in enumerate(line[0]):
        X_test[i][j*len(Alph_Map)+Alph_Map[c]] = 1


from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimators=150000, n_jobs=8)
model.fit(X_train, Y_train)

Y_test = model.predict(X_test)


import csv

with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for y in Y_test:
        writer.writerow([y])
