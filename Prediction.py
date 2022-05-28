import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def prepare_data(dataframe):
    x = dataframe.drop(['final_payment_status'], axis=1)
    y = dataframe['final_payment_status']

    # divide data to test and train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)
    # up-sampling using SMOTE
    smote = SMOTE(sampling_strategy=1, k_neighbors=3, random_state=1)
    x_train_new, y_train_new = smote.fit_resample(x_train, y_train.ravel())

    return x_train_new, y_train_new, x_test, y_test


x_train_new, y_train_new, x_test, y_test = prepare_data(pd.dataframe)
model = RandomForestClassifier()
model.fit(x_train_new, y_train_new)

# make prediction
y_pred = model.predict(x_test)

# Model Evaluation
target_names = ['Unpaid', 'Paid']
print(classification_report(y_test, y_pred, target_names=target_names))
