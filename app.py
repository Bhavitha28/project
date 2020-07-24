import timeit
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
st.title('Credit Card Fraud Detection!')

df=st.cache(pd.read_csv)('fraud.csv')



# Print valid and fraud transactions
fraud=df[df.Class==1]
valid=df[df.Class==0]
outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100

if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
    st.write('Fraud Cases: ',len(fraud))
    st.write('Valid Cases: ',len(valid))

    
#Obtaining X (features) and y (labels)
X=df.drop(['Class'], axis=1)
y=df.Class

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
#Import classification models and metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score


dtree=DecisionTreeClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=0,n_estimators = 10,criterion = 'entropy')
features=X_train.columns.tolist()
etree=ExtraTreesClassifier(random_state=42)


#Feature selection through feature importance
@st.cache
def feature_sort(model,X_train,y_train):
    #feature selection
    mod=model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp


clf=['Decision Trees','Random Forest','Extra Trees']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)

start_time = timeit.default_timer()
if mod_feature=='Decision Trees':
    model=dtree
    importance=feature_sort(model,X_train,y_train)
elif mod_feature=='Random Forest':
    model=rforest
    importance=feature_sort(model,X_train,y_train)
elif mod_feature=='Extra Trees':
    model=etree
    importance=feature_sort(model,X_train,y_train)
elapsed = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60)) 
feature_imp=list(zip(features,importance))
feature_sort=sorted(feature_imp, key = lambda x: x[1])

n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=X_train_sfs
X_test_sfs_scaled=X_test_sfs




from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef


def compute_performance(model, X_train, y_train,X_test,y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',scores
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm  
    cr=classification_report(y_test, y_pred)
    'Classification Report: ',cr
    elapsed = timeit.default_timer() - start_time
    'Execution Time for performance computation: %.2f minutes'%(elapsed/60)

if st.sidebar.checkbox('Run a credit card fraud detection model'):
    
    
    if mod_feature== 'Random Forest':
        model=rforest
        compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            
    elif mod_feature == 'Decision Trees':
        model=dtree
        compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
    elif mod_feature == 'Extra Trees':
        model=etree
        compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            
  


