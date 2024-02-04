# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:01:21 2024

@author: dbda
"""

import pandas as pd
import numpy as np

train=pd.read_csv('train_v9rqX0R.csv')
test=pd.read_csv('test_AbJTz2l.csv')


print(np.sum(train['Item_Weight'].isnull()))
print(train['Item_Fat_Content'].value_counts())


train['Item_Fat_Content'].replace(to_replace='reg',value='Regular',inplace=True)
train['Item_Fat_Content'].replace(to_replace='LF',value='Low Fat',inplace=True)
train['Item_Fat_Content'].replace(to_replace='low fat',value='Low Fat',inplace=True)
test['Item_Fat_Content'].replace(to_replace='reg',value='Regular',inplace=True)
test['Item_Fat_Content'].replace(to_replace='LF',value='Low Fat',inplace=True)
test['Item_Fat_Content'].replace(to_replace='low fat',value='Low Fat',inplace=True)



#concat
trn=train[['Item_Identifier','Item_Weight']]
tst=test[['Item_Identifier','Item_Weight']]

all_item=pd.concat([trn,tst])

# weights1=all_item.groupby('Item_Identifier')['Item_Weight'].mean().reset_index()

# weights2=all_item.drop_duplicates()
# weights2=weights2.dropna()


weights=all_item.dropna()
weights=weights.drop_duplicates()

train.drop('Item_Weight',axis=1,inplace=True)
test.drop('Item_Weight',axis=1,inplace=True)

train=train.merge(weights,how='left',on='Item_Identifier')
test=test.merge(weights,how='left',on='Item_Identifier')

#outlet

outlets=train[['Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']].drop_duplicates()

train['Outlet_Size'].fillna('Small',inplace=True)
test['Outlet_Size'].fillna('Small',inplace=True)

print(np.sum(test.isnull()))
print(np.sum(train.isnull()))

#%%
#encoding the categorical values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer 

ohc = OneHotEncoder(sparse_output=False)


ct = make_column_transformer(('passthrough',
                              make_column_selector(dtype_exclude=object)),
                             (ohc,
                              make_column_selector(dtype_include=object)),
                             verbose_feature_names_out=False).set_output(transform="pandas")

X_train=train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1)
y_train=train['Item_Outlet_Sales']

X_train_trn=ct.fit_transform(X_train)
X_test_trn=ct.transform(test.drop(['Item_Identifier','Outlet_Identifier'],axis=1))

#%%
#XG boost
from xgboost import XGBRegressor
from sklearn.model_selection import KFold,GridSearchCV


x_gbm=XGBRegressor(random_state=23,n_jobs=-1)
kfold=KFold(n_splits=4,shuffle=True,random_state=23)

params = {"learning_rate":[0.001,0.1,0.3],
          'max_depth':[3,5],
          'n_estimators':[25,50]}

gcv_x_gbm=GridSearchCV(x_gbm, param_grid=params,
                 cv=kfold, verbose=3)

gcv_x_gbm.fit(X_train_trn,y_train)
print('params',gcv_x_gbm.best_params_)
print('score',gcv_x_gbm.best_score_)

y_pred_xgbm=gcv_x_gbm.predict(X_test_trn)

ss=test[['Item_Identifier','Outlet_Identifier']]
ss['Item_Outlet_Sales']=y_pred_xgbm
ss.to_csv('Sales_xgb.csv',index=False)


# params {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}
# score 0.6008544653330191

#%%
#XG boost with forest
from xgboost import XGBRFRegressor
from sklearn.model_selection import KFold,GridSearchCV


x_gbm=XGBRFRegressor(random_state=23,n_jobs=-1)
kfold=KFold(n_splits=4,shuffle=True,random_state=23)

params = {"learning_rate":[0.001,0.1,0.3],
          'max_depth':[3,5],
          'n_estimators':[25,50]}

gcv_x_gbm=GridSearchCV(x_gbm, param_grid=params,
                 cv=kfold, verbose=3)

gcv_x_gbm.fit(X_train_trn,y_train)
print('params',gcv_x_gbm.best_params_)
print('score',gcv_x_gbm.best_score_)

y_pred_xgbm=gcv_x_gbm.predict(X_test_trn)

ss=test[['Item_Identifier','Outlet_Identifier']]
ss['Item_Outlet_Sales']=y_pred_xgbm
ss.to_csv('Sales_xgbForest.csv',index=False)


# params {'learning_rate': 0.3, 'max_depth': 5, 'n_estimators': 25}
# score 0.3020503893501891



#%%
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, log_loss

el=ElasticNet()
lr=LinearRegression()
dtc=DecisionTreeRegressor(random_state=23)
rdf=RandomForestRegressor(random_state=23)

# stack=StackingRegressor([('LR',lr),('ELA',el),('TREE',dtc)],
#                          final_estimator=gbm)
stack=StackingRegressor([('LR',lr),('ELA',el),('TREE',dtc)],
                         final_estimator=rdf)


kfold=KFold(n_splits=5,shuffle=True,random_state=23)

params={'ELA__l1_ratio':[0.001,0.2,0.5],
        'ELA__alpha':[0.01,0.5,1],
        'TREE__max_depth':[None,3],
        'final_estimator__max_features':[2,3,4,5],
        'passthrough':[True,False]}
kfold=KFold(n_splits=5,shuffle=True,random_state=23)

gcv_stack=GridSearchCV(stack, param_grid=params,
                 cv=kfold,n_jobs=-1)

gcv_stack.fit(X_train_trn,y_train)
print('best:',gcv_stack.best_params_)

print('score:',gcv_stack.best_score_)


y_pred_stack=gcv_stack.predict(X_test_trn)

ss=test[['Item_Identifier','Outlet_Identifier']]
ss['Item_Outlet_Sales']=y_pred_stack
ss.to_csv('Sales_stack.csv',index=False)

# best: {'ELA__alpha': 1, 'ELA__l1_ratio': 0.001, 'TREE__max_depth': None, 'final_estimator__max_features': 5, 'passthrough': True}
# score: 0.5803439126292098
