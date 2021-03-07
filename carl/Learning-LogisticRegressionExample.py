import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Generate fake data
df = pd.DataFrame(np.random.rand(1000, 4),
                  columns = list('abcd'))
df['Plc'] = np.random.randint(0,2,1000)

# Split X and y
predictors = list('abcd')
X = df[predictors]
y = df['Plc']

# Split train and test
train_size = int(X.shape[0]*0.7)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict train and test
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Stack the prediction and create the column based on the stacked array:
df['preds'] = np.hstack([y_pred_train, y_pred_test])

# or Initialize the column and then assign predictions:
df['preds'] = np.nan
df.loc[:train_size-1, 'pred'] = y_pred_train
df.loc[train_size:, 'pred'] = y_pred_test