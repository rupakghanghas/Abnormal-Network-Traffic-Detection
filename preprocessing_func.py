import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime,timedelta
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import time
import re
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from beautifultable import BeautifulTable
from scipy import stats
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from itertools import product
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix,classification_report,ndcg_score
from sklearn.metrics import roc_auc_score,roc_curve,auc, precision_score, recall_score, precision_recall_curve
from mlxtend.plotting import plot_confusion_matrix
from copy import deepcopy


def change_to_flag(dd,cols): ##flagging some of numeric variables 
    for col in cols:
        dd.loc[:,col]=pd.Series(np.where(dd[col]==0,0,1),index=dd.index)
    return dd

def train_test(dd,x_cols,y_col,stratify_col=None):
    d=dd.copy()
    d=d.sample(frac=1,random_state=0).reset_index(drop=True)
    xtrain,xtest,ytrain,ytest=train_test_split(d.loc[:,x_cols]
                                               ,d.loc[:,y_col]
                                               ,test_size=0.08  #8% test_size
                                               ,random_state=0,stratify=stratify_col) 
    return xtrain,xtest,ytrain,ytest


def outlier_treatment_training(data,col,outlier_dict,target_var,degree_threshold=5):
    df=data.copy()
    percentiles = np.arange(0, 101, 1)

    # Calculate percentiles for the specified column
    percentile_values = np.percentile(df[col], percentiles)

    percentile_df = pd.DataFrame({
        'Percentile': percentiles,
        'Value': percentile_values
    })


    percentile_df['prev_percentile']=percentile_df['Percentile'].shift(periods=1)
    percentile_df['prev_Value']=percentile_df['Value'].shift(periods=1)
    window_size = 5

    # Get global min and max for Percentile and Value
    mean_value = percentile_df['Value'].mean()
    std_value = percentile_df['Value'].std()

    # Function to calculate slope in degrees with global min/max normalization
    def calculate_slope_degrees_normalized(values, percentiles):
        # Normalize x (Percentile) and y (Value) based on global min and max
        x_normalized = percentiles
        y_normalized = (values - mean_value) / (std_value)

        # Calculate slope of the normalized data
        slope, _ = np.polyfit(x_normalized, y_normalized, 1)

        # Convert slope to degrees
        angle_rad = np.arctan(slope)  # Convert slope to radians
        angle_deg = np.degrees(angle_rad)  # Convert radians to degrees
        return angle_deg

    # Apply rolling window to calculate slopes in degrees with normalization
    percentile_df['slope_deg'] = percentile_df['Value'].rolling(window=window_size).apply(
        lambda values: calculate_slope_degrees_normalized(values, percentile_df['Percentile'].iloc[values.index]), 
        raw=False
    )

    # Calculate the difference between consecutive slopes (in degrees) to detect changepoints
    percentile_df['slope_diff_deg'] = percentile_df['slope_deg'].diff()

    # Identify changepoints based on the threshold
    changepoint = percentile_df[(percentile_df['slope_diff_deg'].abs() >= degree_threshold)&\
                               (percentile_df['Percentile']>75)].sort_values('Percentile').prev_Value.iloc[0]
    changepoint_percentile = percentile_df[(percentile_df['slope_diff_deg'].abs() >= degree_threshold)&\
                               (percentile_df['Percentile']>75)].sort_values('Percentile').prev_percentile.iloc[0]
    df.loc[df[col]>changepoint,col]=changepoint
    outlier_dict[col]=changepoint
    print("Changepoints based on slope difference threshold:")
    print(changepoint_percentile,'(percentile) : ',changepoint)
    
    plt.figure(figsize=(12, 6))

    for label in df[target_var].unique():

        sns.kdeplot(df.loc[df[target_var]==label,col], label='{}'.format(label), palette='viridis')
    plt.title('Distribution Plot column: {}'.format(col))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    return df,outlier_dict
    
    
def outlier_treatment_test(data, outlier_dict):
    df = data.copy()
    for col in outlier_dict.keys():
        try:
            cap_value = outlier_dict[col]
            df.loc[df[col] > cap_value, col] = cap_value
        except:
            pass

    return df


def filter_numeric(data,coll):
    enum_coll=list(enumerate(coll))
    cols=data.select_dtypes(include=['number']).columns
    cc=[]
    for col in cols:
        if (data[col].nunique(dropna=True)>2) or\
        ((sorted(list(data[col].dropna().unique()))!=[0,1]) and\
        (sorted(list(data[col].dropna().unique()))!=[0]) and\
        (sorted(list(data[col].dropna().unique()))!=[1])):
            cc.append(col)

    cols=cc
    numeric_cols=list(set(cols).intersection(set(coll)))
    li=[sett[1] for sett in enum_coll if sett[1] in numeric_cols]
    return li

def standardize_data(d,cols,strategy='standard'):
    if strategy=='standard':
        scalar=StandardScaler()
    else:
        scalar=MinMaxScaler()
    data=d.copy()
    cols=filter_numeric(data,cols)
    #print(cols)
    data.loc[:,cols]=pd.DataFrame(scalar.fit_transform(data[cols]),columns=cols,index=data.index)

    return data,scalar,cols

def transform_standardize_data(d,cols,scalar):
    data=d.copy()
    data.loc[:,cols]=pd.DataFrame(scalar.transform(data[cols]),columns=cols,index=data.index)
    return data

def revtransform_standardize_data(d,cols,scalar):
    data=d.copy()
    data.loc[:,cols]=pd.DataFrame(scalar.inverse_transform(data[cols]),columns=cols,index=data.index)
    return data

## One hot encoding
def dummy_str_col(d,cols):
    data=d.copy()
    drop_str_cols=[]
    str_cols=cols
    for str_col in str_cols:
        str_data=pd.get_dummies(data.loc[:,str_col]).astype(int)
        str_data.columns=[str_col+'_'+str(col) for col in str_data.columns]
        drop_str_cols.append(str_data.columns[-1])
        str_data.drop(str_data.columns[-1],axis=1,inplace=True)

        data.drop(str_col,axis=1,inplace=True)
        data=pd.concat([data,str_data],axis=1)
    return data,drop_str_cols

def test_str_col(d,cols,drop_str_cols):
    data=d.copy()
    str_cols=cols
    for str_col in str_cols:
        str_data=pd.get_dummies(data.loc[:,str_col]).astype(int)
        str_data.columns=[str_col+'_'+str(col) for col in str_data.columns]
        data.drop(str_col,axis=1,inplace=True)
        data=pd.concat([data,str_data],axis=1)

    for drop_col in drop_str_cols:
        data.drop(drop_col,axis=1,inplace=True)
        
    return data

# WoE encoding function
def calculate_woe(data, feature, target_var):
    # Group by the feature to calculate WoE
    temp_df = data.groupby(feature)[target_var].agg(['count', 'sum'])
    temp_df['non_event'] = temp_df['count'] - temp_df['sum']
    temp_df['event_dist'] = temp_df['sum'] / temp_df['sum'].sum()
    temp_df['non_event_dist'] = temp_df['non_event'] / temp_df['non_event'].sum()
    temp_df['woe'] = np.log(temp_df['event_dist'] / temp_df['non_event_dist']).replace([np.inf, -np.inf], 0)
    
    # Return a mapping of WoE values for each category
    return temp_df['woe']

# Training function
def woe_encoding_train(df, to_encode_cols, target_var):
    d = df.copy()
    encoding_dict = {}

    for col in to_encode_cols:
        woe_values = calculate_woe(d, col, target_var)
        encoding_dict[col] = woe_values.to_dict()
        
        # Apply WoE transformation
        d[col + '_woe'] = d[col].map(encoding_dict[col]).fillna(0)
    
    return d, encoding_dict

# Test function
def woe_encoding_test(df, encoding_dict):
    d = df.copy()
    
    for col, woe_map in encoding_dict.items():
        # Assign 0 as the fallback WoE value for unseen categories
        d[col + '_woe'] = d[col].map(woe_map).fillna(0)
    
    # Drop original columns after encoding
    d.drop(columns=encoding_dict.keys(), inplace=True)
    
    return d