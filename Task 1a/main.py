from sklearn.metrics import mean_squared_error

def evaluate(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5

from sklearn.linear_model import Ridge
import numpy as np

# Read the data and remove the first Row
train_data = np.genfromtxt('train.csv', delimiter = ',') 
train_data = train_data[1:]

# Make an array of data arrays X = [[x00,x01,...,x09],...]
X = np.array([row[1:] for row in train_data])

# Build Vector for Result y = [y0,y1,...,yn]
y = np.array([row[0] for row in train_data])

# Cross-Validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
kf.get_n_splits(X)

alphas = [0.1,1,10,100,200]
errors = [0] * 5

for i, alpha in enumerate(alphas, 0):
    y_results_predicted = []
    y_results_real = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reg = Ridge(alpha=alpha).fit(X_train, y_train)
        y_predicted = reg.predict(X_test)
        y_results_predicted.append(y_predicted)
        y_results_real.append(y_test)
    errors[i] = evaluate(y_results_real, y_results_predicted)

print(errors)
      

#print(reg.score(X_train, y_train))
# Predict the Results
#y_predicted = reg.predict(X_test)
#print(y_predicted)
