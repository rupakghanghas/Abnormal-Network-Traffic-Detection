import streamlit as st
import pandas as pd
import datetime
import preprocessing_func as pf
import pickle
import joblib

st.set_page_config(layout="wide")

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

# Create a form
with st.form(key='network_anamoly'):
    st.write("## Abnormal-Network-Traffic-Detection")
    st.write("### Main Features")
    # First row
    col1, col2 = st.columns(2)

    with col1:
        lastflag = int(st.slider("Last Flag", 0, 21, step=1))
        flag=st.selectbox("Flag", ('OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH'))
        
        dstbytes = float(st.text_input("DST Bytes", 0))
        srcbytes = float(st.text_input("SRC Bytes", 0))

    with col2:
        diffsrvrate=float(st.slider("Different Service Rate(%) (in past 2 secs)", 0.0, 1.0, step=0.01))
        countt=float(st.text_input("Same Destination Host Connections (in past 2 secs)", 0))
        dsthostsrvcount=float(st.text_input("Same Service Connections (whole timeline)", 0))
        dsthostdiffsrvrate=float(st.slider("Different Service Rate(%) (whole timeline)", 0.0, 1.0, step=0.01))


    
    st.write("### Additional Features")
    # Second row
    col4, col5, col6 = st.columns(3)

    with col4:
        suattempted=int(st.slider("'su root' Command Attempted (0/1)", 0, 1, step=1))
        rootshell=int(st.slider("Root Shell Obtained (0/1)", 0, 1, step=1))
        numfailedlogins=int(st.slider("Failed Login (0/1)", 0, 1, step=1))
        land = int(st.slider("SRC and DST IP addresses and port numbers Equal (0/1)", 0, 1, step=1))
        numfilecreations = int(st.slider("File Creation operations Applied (0/1)", 0, 1, step=1))

    with col5:
        wrongfragment = int(st.slider("Wrong Fragment in connection (0/1)", 0, 1, step=1))
        urgent = int(st.slider("Urgent packet in connection (0/1)", 0, 1, step=1))
        srvdiffhostrate = float(st.slider("Diff Host Same Service Rate(%) (in past 2 secs)", 0.0, 1.0, step=0.01))
        protocoltype = st.selectbox("Protocol Type", ('tcp', 'udp', 'icmp'))
        isguestlogin = int(st.slider("Guest Login (0/1)", 0, 1, step=1))

    with col6:
        dsthostsrvdiffhostrate = float(st.slider("Different Host Same Service Rate(%) (whole timeline)", 0.0, 1.0, step=0.01))
        dsthostcount = int(st.text_input("Same Host Connections (whole timeline)", 100))
        numaccessfiles = int(st.slider("Operation on Access Control Files (0/1)", 0, 1, step=1))
        srvcount = int(st.text_input("Same Service Connections (in past 2 secs)", 8))
        numcompromised = int(st.slider("Compromised Condition (0/1)", 0, 1, step=1))
        
    # Third row for additional float features
    st.write("### More Additional Features")
    col7, col8 = st.columns(2)

    with col7:
        dsthostsamesrcportrate = float(st.slider("Same Host and Same Service Rate(%) (whole timeline)", 0.0, 1.0, step=0.01))
        dsthostsrvrerrorrate = float(st.slider("Same Service REJ Error Rate(%) (whole timeline)", 0.0, 1.0, step=0.01))
        numroot = int(st.slider("Root Access/Operations performed as Root (0/1)", 0, 1, step=1))
        

    with col8:
        duration = int(st.text_input("Connection Duration (seconds)", 20))
        hot = int(st.slider("Hot indicator in content (0/1)", 0, 1, step=1))
        numshells = int(st.slider("Shell prompt (0/1)", 0, 1, step=1))

    # Submit button
    submit_button = st.form_submit_button(label="Submit")

    
if submit_button:
    # Create a dictionary with all the entered values
    data = {
        # Important features
        'lastflag': [lastflag],
        'flag': [flag],
        'dstbytes': [dstbytes],
        'srcbytes': [srcbytes],
        'diffsrvrate': [diffsrvrate],
        'countt': [countt],
        'dsthostsrvcount': [dsthostsrvcount],
        'dsthostdiffsrvrate': [dsthostdiffsrvrate],

        # Non-important features
        'suattempted': [suattempted],
        'rootshell': [rootshell],
        'numfailedlogins': [numfailedlogins],
        'land': [land],
        'numfilecreations': [numfilecreations],
        'wrongfragment': [wrongfragment],
        'urgent': [urgent],
        'srvdiffhostrate': [srvdiffhostrate],
        'protocoltype': [protocoltype],
        'isguestlogin': [isguestlogin],
        'dsthostsrvdiffhostrate': [dsthostsrvdiffhostrate],
        'dsthostcount': [dsthostcount],
        'numaccessfiles': [numaccessfiles],
        'srvcount': [srvcount],
        'numcompromised': [numcompromised],
        'dsthostsamesrcportrate': [dsthostsamesrcportrate],
        'dsthostsrvrerrorrate': [dsthostsrvrerrorrate],
        'numroot': [numroot],
        'duration': [duration],
        'hot': [hot],
        'numshells': [numshells]
    }

    # Create a DataFrame from the dictionary
    
    df = pd.DataFrame(data)

    # Display the DataFrame
    st.write("### Output Based on Submitted Data")
    #st.write(str({d:data[d][0] for d in data.keys()}))
    
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
    
    #st.dataframe(df)
    display_text = r'<span style="font-size:20px;">Probability that the connection is:<br>'

    y_pred_probs=y_pred_probs.iloc[0].sort_values(ascending=False)
    for ind in y_pred_probs.index:
        # Format the column name in red and uppercase, and percentage in bold red
        display_text += r'<span style="color:red;font-size:15px;">{} : </span> <span style="color:red; font-size:15px; font-weight:bold;">{:.1f}%</span><br>'.format(ind.title(), float(y_pred_probs[ind]) * 100)

    st.markdown(display_text+r'</span>', unsafe_allow_html=True)
    
    
