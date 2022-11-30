import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import warnings

warnings.filterwarnings('ignore')
# LGBM Model
import lightgbm as lgb

# Load the csv file
df_recommended = pd.read_csv('Cropping_System_final_Model.csv')

print(df_recommended.head())

# Select independent and dependent variable
X = df_recommended.drop('Label', axis=1)
y = df_recommended['Label']

# Split the dataset into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=50)

# Instantiate the model
lgbm_model = lgb.LGBMClassifier()

# Fit the model
lgbm_model.fit(X_train, y_train)
#y_pred = lgbm_model.predict(X_test)
#print("LightGBM Model accuracy score is :{0:0.4f}.".format(accuracy_score(y_pred, y_test)))

# Fit the model
# classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(lgbm_model, open("LGBMModel.pkl", "wb"))
