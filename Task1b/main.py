import numpy as np
import math

def makeTransformation(row):
    return np.concatenate((row, list(map((lambda x: x*x), (row))),list(map((lambda x: math.exp(x)), (row))),list(map((lambda x: math.cos(x)), (row))),[1]))

# Read the data and remove the first Row
train_data = np.genfromtxt('train.csv', delimiter = ',') 
train_data = train_data[1:]

# Make an array of data arrays X = [[x00,x01,...,x09],...]
X = np.array([row[2:] for row in train_data])

# Build Vector for Result y = [y0,y1,...,yn]
y = np.array([row[1] for row in train_data])

from sklearn.linear_model import LinearRegression

X_train = [makeTransformation(row) for row in X]
    
print(X_train)

reg = LinearRegression(fit_intercept=False).fit(X_train, y)

print(reg.coef_)


import csv

with open('results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for coef in reg.coef_:
        writer.writerow([coef])