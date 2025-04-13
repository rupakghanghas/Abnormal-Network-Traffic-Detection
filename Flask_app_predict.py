from flask import Flask, request, jsonify
#FLASK_APP=loan_app_flask.py flask run
app=Flask(__name__)
print(__name__)

@app.route("/")
def hello_world():
    return "<p> Hello World!</p>"



import pandas as pd
import datetime
import preprocessing_func as pf
import pickle
import joblib

param_dict=pickle.load(open('artifacts.pkl','rb'))
#param_grid=param_dict['param_grid']
checkpoint_index=param_dict['checkpoint_index']  ##one already covered
best_param_index=param_dict['best_param_index']
best_estimator=joblib.load("bestmodel.joblib")

best_val_accuracy=param_dict['best_val_accuracy']
best_val_ndcg=param_dict['best_val_ndcg']


outlier_dict=param_dict['outlier_dict']
drop_cols=param_dict['drop_cols']
encoding_dict=param_dict['encoding_dict']
numeric_cols=param_dict['numeric_cols']
scalar=param_dict['scalar']
target_encode=param_dict['target_encode']
rev_target_encode=param_dict['rev_target_encode']

pca=param_dict['pca']
non_imp_features_transformed=param_dict['non_imp_features_transformed']
imp_features_transformed=param_dict['imp_features_transformed']


@app.route("/predict",methods=['POST'])
def prediction():
    network_req=request.get_json()
    df = pd.DataFrame(network_req, index=[0])

    df=pf.outlier_treatment_test(df, outlier_dict)

    df['protocoltype_icmp']=0
    df['protocoltype_tcp']=0
    df['protocoltype_udp']=0
    df['protocoltype_'+df.loc[0,'protocoltype']]=1
    df.drop(drop_cols+['protocoltype'],axis=1,inplace=True)

    df=pf.woe_encoding_test(df, encoding_dict)
    df=pf.transform_standardize_data(df,numeric_cols,scalar)
    
    X_pca_cleaned = pca.transform(df[non_imp_features_transformed])
    xtest_pca=pd.DataFrame(X_pca_cleaned,columns=['col_{}'.format(j) for j in range(X_pca_cleaned.shape[1])])
    df=pd.concat([df[imp_features_transformed],xtest_pca],axis=1)
    
    y_pred_probs=pd.DataFrame(best_estimator.predict_proba(df),columns=best_estimator.classes_)
    y_pred_probs.columns=[rev_target_encode[col] for col in y_pred_probs.columns]

    y_pred_probs=y_pred_probs.iloc[0].sort_values(ascending=False)
    return_dictt={}
    for ind in y_pred_probs.index:
        return_dictt['{} probability(%)'.format(ind.upper())]=round(float(y_pred_probs[ind]) * 100,2)
        

    return return_dictt