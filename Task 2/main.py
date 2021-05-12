import numpy as np

# Read the data and remove the first Row
train_data = np.genfromtxt('train_features.csv', delimiter = ',') 
train_data = train_data[1:]

# Make an array of data arrays X = [[x00,x01,...,x09],...]
X_all = np.array([row[2:] for row in train_data])
X_subgroups = np.array_split(X_all, len(X_all)/12)
X_avg = np.mean(X_subgroups, axis=0)

# Build Vector for Result y = [y0,y1,...,yn]
y = np.array([row[1] for row in train_data])





""" import csv

with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for coef in reg.coef_:
        writer.writerow([coef]) """

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