🔐 Network Anomaly Detection using KDD’99 Dataset

A machine learning-based intrusion detection system (IDS) that classifies network traffic into normal or malicious activity using the benchmark KDD’99 dataset. This project applies advanced preprocessing, feature engineering, and classification techniques to detect various types of cyber attacks.
📌 Table of Contents

    📖 Description

    ⚙️ Working

    📊 Dataset Details

    🚀 Features

    🧠 Model Pipeline

    📈 Results

    🔍 Future Work

    📂 Folder Structure

    💻 How to Run

    🤝 Contributing

    📜 License

📖 Description

Cyber attacks are increasingly sophisticated, and traditional signature-based intrusion detection systems fall short when facing novel threats. This project tackles this challenge by using machine learning to classify network connections as normal or belonging to one of four attack types: DoS, Probe, R2L, and U2R.

The solution involves data cleaning, feature engineering, class imbalance handling, and the use of various anomaly detection and classification models.
⚙️ Working

The working pipeline includes:

    Loading the KDD’99 dataset

    Preprocessing:

        Handling nulls, duplicates, and irrelevant data

        Label mapping and balancing classes using SMOTE

    Feature Engineering:

        ANOVA, Chi-Square, Kruskal-Wallis tests

        WoE encoding, one-hot encoding

        Outlier detection using Isolation Forest and LOF

        VIF for multicollinearity reduction

        Dimensionality reduction using PCA

    Model Training:

        Trained classifiers like Random Forest, XGBoost, and Logistic Regression

    Evaluation:

        Accuracy, precision, recall, F1-score

        Confusion matrix and classification report

📊 Dataset Details

    Dataset: KDD Cup 1999

    Records: 4,898,431 total instances

    Features: 41

    Label: Normal or one of 22 attack types

    Grouped Labels:

        DoS (Denial of Service)

        Probe

        R2L (Remote to Local)

        U2R (User to Root)

🚀 Features

     Advanced preprocessing pipeline

     Handles class imbalance using SMOTE

     Applies feature selection via statistical tests

     Reduces dimensionality using PCA

     Uses ensemble models for higher accuracy

     Visualizes results with matplotlib/seaborn

🧠 Model Pipeline

    Models used:

    Random Forest

    Logistic Regression

    XGBoost

    Isolation Forest

    Local Outlier Factor (LOF)

    Evaluation metrics:

    Accuracy

    Precision, Recall, F1-Score

    Confusion Matrix

    ROC-AUC Curve

📈 Results
Model	Accuracy	F1-Score	Precision	Recall
Random Forest	98.7%	0.987	0.988	0.987
XGBoost	98.5%	0.985	0.986	0.985
Logistic Regression	94.2%	0.943	0.945	0.941

    Note: Results may vary slightly depending on train-test splits and random seed.

🔍 Future Work

    Integrate deep learning models (e.g., LSTM, Autoencoders)

    Deploy the model as a real-time monitoring tool using Flask/FastAPI

    Extend analysis to NSL-KDD, UNSW-NB15, or CIC-IDS2017

    Integrate active learning to continuously adapt to new threats
    
💻 How to Run
    
    # Clone the repository
    git clone https://github.com/yourusername/network-anomaly-detection.git
    cd network-anomaly-detection
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Run the project
    python Streamlit_predict.py
🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit a PR.
📜 License

This project is licensed under the MIT License.
  
