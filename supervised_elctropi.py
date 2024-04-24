#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
Y = pd.DataFrame(target, columns=["MEDV"])

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    features = {}
    for feature in X.columns:
        features[feature] = st.sidebar.slider(feature, X[feature].min(), X[feature].max(), X[feature].mean())
    return pd.DataFrame(features, index=[0])

df = user_input_features()

st.header('Specified Input parameters')
st.write(df)
st.write('---')

model = RandomForestRegressor()
model.fit(X, Y)

prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')


# In[ ]:




