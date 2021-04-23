from sklearn.metrics import mean_squared_error

def evaluate(y, y_pred):
    RMSE = mean_squared_error(y, y_pred)**0.5
    print(RMSE)

import numpy as np
from sklearn.linear_model import LinearRegression

# Read the Training and Test data
train_data = np.genfromtxt('train.csv', delimiter = ',')
test_data = np.genfromtxt('test.csv', delimiter = ',')

# Remove First Row of non-data 
train_data = train_data[1:]
test_data = test_data[1:]

# Make an array of data arrays X = [[x00,x01,...,x09],...]
tmp = []
for row in train_data:
    # "2:" as the format is ID,y,x1,x2,...
    tmp.append(row[2:])
X_train = np.array(tmp)

tmp = []
for row in test_data:
    # "1:" as the format is ID,x1,x2,...
    tmp.append(row[1:]) 
X_test = np.array(tmp)

# Build Vector for Result y = [y0,y1,...,yn]
w = []
for row in train_data:
    w.append(row[1])
y_train = np.array(w)

# Fit a Function to the data using X and y: c + a0 * x0 + a1 * x1 + ... 
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_train, y_train))
print(reg.coef_)

# Predict the Results
y_predicted = reg.predict(X_test)
print(y_predicted)

# Create np Array for result file
results = []
for id,y in zip(test_data,y_predicted):
    results.append([int(id[0]),y])

import csv

with open('results.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id","y"])
    for i in range (len(results)):
        writer.writerow(results[i])