# Packages importing
#%%
import numpy as np
import pandas as pd
import tkinter
import matplotlib
# matplotlib.use('TkAgg')  # !IMPORTANT
import matplotlib.pyplot as plt
import seaborn as sns
import scipy, os
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
random_seed=123
target_names = ['Not Deprived', 'Deprived']
THRESHOLD= 0.8

#%%
# Helper function
from toolbox import get_train_val_ps, important_55, cm, calculate_vif, covariate_features, contextual_features
covariate_features= covariate_features()
contextual_features= contextual_features()
best_55= important_55()
rgbn= ['r','g','b','n']
#%%
# Load Data into Pandas Dataframe
accra1= pd.read_parquet('accra_covariate_100m_df.parquet.gzip').drop(['labels','type'],axis=1)
print(f'Null values: {accra1.isna().sum().any()}')
accra2= pd.read_parquet('accra_contextual_100m_df.parquet.gzip')
print(f'Null values: {accra2.isna().sum().any()}')
accra3= pd.read_parquet('accra_rgb_100m_df.parquet.gzip').drop(['labels','type'],axis=1)
print(f'Null values: {accra3.isna().sum().any()}')

#Concatenate the 3-feature dataset (rgbn,cov,contx)
accra= pd.concat([accra3,accra1,accra2],axis=1)
accra= accra[~(accra==-9999.000000)]
# Check for Nulls $ replace with mean
print(accra.isna().sum().sort_values(ascending=False))
accra.dropna(subset=['labels'],inplace=True) #drop nulls with labels column with nulls
accra['labels']= accra.labels.astype('int') # Change labels to integers
##
best_55.append('labels')
best_55.append('type')
accra_copy= accra.loc[:,accra.columns.isin(best_55)].copy(deep=True)
values= {i:accra_copy[i].mean() for i in accra_copy.columns[:-2]}
accra_copy.loc[:,accra_copy.columns[:-2]]= accra_copy.loc[:,accra_copy.columns[:-2]].fillna(value=values)

# accra_copy.fillna(0,inplace=True)

accra_copy.to_parquet('accra_areas_combined.parquet.gzip',compression='gzip')
#%%
# Split Train, Validation & Test
train= accra_copy[accra_copy.type=='train'].copy(deep=True).drop('type',axis=1)
test= accra_copy[accra_copy.type=='test'].copy(deep=True).drop('type',axis=1)
validation= accra_copy[accra_copy.type=='validation'].copy(deep=True).drop('type',axis=1)

#%%
# LOGISTIC REGRESSION & RANDOM FOREST (55 FEATURES)
best_55.remove('type')
train_c= train.loc[:,train.columns.isin(best_55)]
val_c= validation.loc[:,validation.columns.isin(best_55)]
test_c= test.loc[:,test.columns.isin(best_55)]
# Split into features and target
x_train, y_train= train_c.iloc[:,:-1], train_c.iloc[:,-1].values
x_val, y_val= val_c.iloc[:,:-1], val_c.iloc[:,-1].values
x_test, y_test= test_c.iloc[:,:-1], test_c.iloc[:,-1].values
print(x_train.shape,x_val.shape,x_test.shape)

# Target Balance Check
sns.barplot(x=train_c.labels.value_counts().index,y=train_c.labels.value_counts())
plt.xticks(ticks=[0,1],labels=['Not Deprived','Deprived'])
plt.show()

#%%
# models
models = {'lr': LogisticRegression(class_weight='balanced', random_state=random_seed),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=random_seed),
          'xgbc':XGBClassifier(random_state=random_seed)}
# Scale features
mms =MinMaxScaler()
# Standardize the training, val & test feature data
x_train = mms.fit_transform(x_train)
x_val = mms.transform(x_val)
x_test = mms.transform(x_test)

# Creating Dictionary for Pipeline
from sklearn.pipeline import Pipeline
pipes={}
for acronym, model in models.items():
  pipes[acronym]= Pipeline([('model', model)])

# Getting the predefined split cross-validator
# Get the:
# feature matrix and target vector in the combined training and validation data
# target vector in the combined training and validation data
# PredefinedSplit

x_train_val, y_train_val, ps = get_train_val_ps(x_train, y_train, x_val, y_val)

from scipy.stats import uniform

param_dists = {}

# FOR LOGISTIC REGRESSION
# The distribution for tol_grid: a uniform distribution over [loc, loc + scale]
tol_grid = uniform(loc=0.000001, scale=15)

# The distribution for C_grid & solver: a uniform distribution over [loc, loc + scale]
C_grid = uniform(loc=0.001, scale=10)
solver = ['lbfgs', 'sag']
penalty=['l1','l2']

# Update param_dists
param_dists['lr'] = [{'model__tol': tol_grid,
                      'model__C': C_grid, 'model__solver': solver}]

# ============
# FOR RandomForestClassifier
# The distribution for n_estimator,: a uniform distribution over [loc, loc + scale]

min_samples_split = [2, 20, 100]
min_samples_leaf = [1, 20, 100]
n_estimators= [10,30,100,200,500]
max_depth= [10,20,30]
# Update param_dists
param_dists['rfc'] = [{'model__min_samples_split': min_samples_split,
                       #'model__n_estimators':n_estimators,
                       'model__min_samples_leaf': min_samples_leaf,
                       'model__max_depth':max_depth}]
# FOR XGBOOST
n_estimators= [10,30,100,200,500]
max_leaves= [10,15,20,25,30]
eval_metric= ['logloss','error']
objective= ['binary:logistic']
param_dists['xgbc']=[
    {'model__max_depth':max_depth,
     'model__max_leaves':max_leaves,
     'model__n_estimators':n_estimators,
     'model__eval_metric':eval_metric,
     'model__objective':objective}
]

# ============

from sklearn.model_selection import RandomizedSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by RandomizedSearchCV
best_score_params_estimator_rs = []

for acronym in pipes.keys():
    # RandomizedSearchCV
    rs = RandomizedSearchCV(estimator=pipes[acronym],
                            param_distributions=param_dists[acronym],
                            n_iter=2,
                            scoring='f1_macro',
                            n_jobs=2,
                            cv=ps,
                            random_state=random_seed,
                            return_train_score=True)

    # Fit the pipeline
    rs = rs.fit(x_train_val, y_train_val)

    # Update best_score_param_estimators
    best_score_params_estimator_rs.append([rs.best_score_, rs.best_params_, rs.best_estimator_])

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(rs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score',
                         'std_test_score',
                         'mean_train_score',
                         'std_train_score',
                         'mean_fit_time',
                         'std_fit_time',
                         'mean_score_time',
                         'std_score_time']

    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    # cv_results.to_csv(acronym + '96.csv',index=False)

# Sort best_score_params_estimator_rs in descending order of the best_score_
best_score_params_estimator_rs = sorted(best_score_params_estimator_rs, key=lambda x: x[0], reverse=True)

#%%
# Print best_score_params_estimator_rs
print(pd.DataFrame(best_score_params_estimator_rs, columns=['best_score', 'best_param', 'best_estimator']))
print(cv_results['params'][0])
print(best_score_params_estimator_rs)

best_score_gs, best_params_gs, best_estimator_gs = best_score_params_estimator_rs[0]

#Save Model
import pickle
with open('model_lr_55_accra.pickle','wb') as f:
    pickle.dump(best_estimator_gs, f)

# Get the prediction on the test data using the best model
y_test_pred = best_estimator_gs.predict(x_test)
# y_test_pred= pd.Series(best_estimator_gs.predict_proba(x_test)[:,1]).apply(lambda x:0 if x>THRESHOLD else 1)

cm(y_test, y_test_pred) #Confusion matrix
print(classification_report(y_test, y_test_pred, target_names=target_names))

