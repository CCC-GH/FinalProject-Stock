# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit
from sklearn.linear_model import LinearRegression
import numpy as np

x = [1,2,3,4,5,6,7]
y = [1,2,1,3,2.5,2,5]

# Shape of (7,1) for input x and (7,) or (7,1) for target y.
X = np.array(x).reshape(-1, 1)
y = np.array(y)          # You may omit this step if you want

# Create linear regression object
regr = LinearRegression()

regr.fit(X, y)          
# Prediction Time
X_new = np.array([1, 2000, 3, 4, 5, 26, 7]).reshape(-1, 1)
print(regr.predict(X_new))

print(regr.predict([[2000]]))

# X : numpy array or sparse matrix of shape [n_samples,n_features]
#   Training data

# y : numpy array of shape [n_samples, n_targets]
#   Target values. Will be cast to X’s dtype if necessary

# When you do [[2000]], it will be internally converted to np.array([[2000]]), 
# so it has the shape (1,1). This is similar to (n_samples, n_features), 
# where n_features = 1. This is correct for the model because at the training,
# the data has shape (n_samples, 1). So this works.

# Now lets say, you have:

# X_new = [1, 2000, 3, 4, 5, 26, 7] #(You havent wrapped it in numpy array and reshape(-1,1)
# yet again, it will be internally transformed as this:

# X_new = np.array([1, 2000, 3, 4, 5, 26, 7])
# So now X_new has a shape of (7,). See its only a one dimensional array. 
# It doesn't matter if its a row vector or a column vector. Its just one-dimensional array of (n,).

# So scikit may not infer whether its n_samples=n and n_features=1 or other way
# around (n_samples=1 and n_features=n). Please see my other answer which explains about this.

# So we need to explicitly convert the one-dimensional array to 2-d by reshape(-1,1).