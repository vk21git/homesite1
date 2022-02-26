import pandas as pd
import numpy as np
import pickle
import os
import lightgbm as gbm
from sklearn import model_selection
from sklearn import linear_model ,metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn import model_selection,metrics


os.chdir("static/files")

RFE_columns = pd.read_csv('RFE_features_1.csv').columns
#print(RFE_columns)
df = pd.read_csv('train.csv')
#RFE_columns = [col for col in RFE_columns if col not in 'QuoteConversion_Flag']  # Test wont have the label column

#    RFE_columns = [col for col in RFE_columns if col not in 'QuoteConversion_Flag']  # Test wont have the label column
df_RFE = df[RFE_columns]
df_RFE.drop(columns=['Original_Quote_Date', 'SalesField8'], axis=1, inplace=True)
#print(df_RFE.to_string())

## Dropping few columns from df_mutual as they have to many catergories as we are going to model the annomymus feature
## columns as purely catergorical







# # Load the test set
# df = pd.read_csv("sample.csv")
# print(df.to_string())


#df_RFE = df.loc[:, RFE_columns]
#print(df.to_string())
## just take the required columns requied for predicting on it



## Parameters for Light GBM using Reculsive Feature Elimination
RFE_params = {
    'boosting_type': 'gbdt',
    'lambda_l1': 4.540006226304331e-08,
    'lambda_l2': 4.715716309514142,
    'num_leaves': 105,
    'feature_fraction': 0.89,
    'bagging_fraction': 1,
    'bagging_freq': 4,
    'min_child_samples': 65,
    'max_bin': 20,
    'learning_rate': 0.14, }

### Parameters for Light GBM using Mutual info
mutual_info_params = {'boosting_type': 'gbdt',
                      'lambda_l1': 4.956734949314487e-08,
                      'lambda_l2': 2.278541145546624e-08,
                      'num_leaves': 131,
                      'feature_fraction': 0.6,
                      'bagging_fraction': 0.76,
                      'bagging_freq': 2,
                      'min_child_samples': 21,
                      'max_bin': 18,
                      'learning_rate': 0.15}

## Intialize the models
RFE_gbm = gbm.LGBMClassifier(**RFE_params)
mutual_gbm = gbm.LGBMClassifier(**mutual_info_params)



X_train, X_val, y_train, y_val = model_selection.train_test_split(df_RFE.drop('QuoteConversion_Flag', axis=1),
                                                                  df_RFE['QuoteConversion_Flag'], random_state=42,
                                                                  stratify=df_RFE['QuoteConversion_Flag'])


GBM2 = Pipeline([('label_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-99)),
    ('RFE_gbm', RFE_gbm)])

GBM2.fit(X_train, y_train)
y_predict_RFE = GBM2.predict_proba(X_val)

print(y_predict_RFE[:, 1])

## Validation RFE AUC
print("AUC score for the Light GBM RFE ensemble is:{:.2f}".format(metrics.roc_auc_score(y_val, y_predict_RFE[:, 1])))

# save the model to disk
filename = 'vikas_model.pkl'
pickle.dump(GBM2, open(filename, 'wb'))
