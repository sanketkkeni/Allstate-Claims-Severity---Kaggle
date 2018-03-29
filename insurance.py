import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import Ridge
from playsound import playsound
def audio():
    playsound('C:\\Users\\Sanket Keni\\Music\\notification.mp3')
from slackclient import SlackClient
def slack_message(message, channel):
    token = 'xoxp-332552408416-334046357094-332656351905-d0acfa81d2cc0be0e3dfba213ad36291'
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel, 
                text=message, username='My Sweet Bot',
                icon_emoji=':robot_face:')





test_data = pd.read_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\insurance\\test.csv")
train_data = pd.read_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\insurance\\train.csv")

complete = pd.DataFrame(train_data.append(test_data))

complete = complete.drop('id', axis = 1)

print(complete.head(10)); 

#Displaying all the columns in describe by setting display maximum columns
pd.set_option('display.max_columns', None)
print(complete.describe(include = ['O'])) # categorical variables
print(complete.describe()) # contimuous variables

#Checking the skew of the train_data
print(complete.skew())

#Checking the kurtosis of the train_data
complete.kurt()

cont_vars = complete.iloc[:,116:130].columns.values
print(len(cont_vars))
cat_vars = complete.iloc[:,:116].columns.values
print(len(cat_vars))

#sn.boxplot(train_data["cont1"])
plt.clf()
plt.figure(figsize=(12,9))
sn.boxplot(data = complete[cont_vars])

plt.figure(figsize=(6,6))
sn.boxplot(y = complete['loss'], width = 0.8, color = 'yellow')
plt.show()

# Log transforming to bring into normal form
complete.iloc[:,116:] = complete.iloc[:,116:].transform(lambda x: np.log1p(x))

print("*SKEW*")
print("")
print(complete.skew())
print("")
print("*KURT*")
print("")
print(complete.kurt())

for i in range(len(cont_vars)):
    plt.figure(figsize=(8,8))
    sn.distplot(a = (complete[cont_vars[i]]), kde = True, color = 'blue')
    
loss_col = complete['loss']
plt.figure(figsize=(8,8))
sn.distplot(a = (loss_col[:188318]), color = 'blue')

(complete.corr())

plt.figure(figsize = (14,8))
sn.heatmap(complete.corr(), cmap = 'Blues', linewidths = 0.2)

print(test_new.shape)

# Creating dummy variables for the categorical features for further analysis
complete = pd.get_dummies(complete.iloc[:,:])
print(complete.head(2))
print(complete.shape)

#188318, 132
train_new = complete.iloc[:188318,:]
print(train_new.shape)
test_new = complete.iloc[188318:,:]
test_new = test_new.drop('loss', axis = 1)
print(test_new.shape)

#Separating predictor and target features
from sklearn.model_selection import train_test_split

y = train_new['loss']
y = y.reshape(train_new.shape[0], 1) # convert to nd numpy array
print(type(y))
X = train_new.drop(['loss'], axis = 1)
X= X.as_matrix() # convert to nd numpy array
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 20)
audio()

print(X_train.shape)
print(X_val.shape)
print(y_train.shape)


# LASSO for feature selection
from sklearn.metrics import mean_absolute_error as mae, r2_score as rs
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

lasso1 = Lasso(alpha = 0.001, random_state = 20) # increasing alpha reduces the number of features obtained
lasso1.fit(X_train,y_train)
lasso_coeff = lasso1.coef_    #Finding the important coefficients.
index = []
for i,val in enumerate(lasso_coeff):
    if val!=0:
        index.append(i)
audio()
X_train = X_train[:,index]                    #Selecting only important features in the train set
print(X_train.shape)

X_val = X_val[:,index]                     #Selecting only important features in the validation set
X_val.shape
test_new = test_new.iloc[:,index]
test_new.shape

l1 = lasso1.fit(X_train, y_train)
result = mae(np.expm1(y_val), np.expm1(lasso1.predict(X_val)))          #Finding the mean absolute error
print(result)      #MAE = 1273.7017847938228




#Ridge

alpha = list([0.2,0.4])
rid_reg = Ridge(random_state = 20)
param_grid = {'alpha': alpha}

rid_cv = GridSearchCV(rid_reg, param_grid, verbose = 1, cv = 3)      #Applying gridsearch for ridge regression
rid_cv.fit(X_train,y_train)

print(rid_cv.best_params_)  #Best alpha - {'alpha': 2.5}
rid_cv.best_score_   #Best R-square value - 0.51087109218284943

result = mae(np.expm1(y_val), np.expm1(rid_cv.predict(X_val)))
print(result)   #MAE = 1274.558807625875
audio()


# LINEAR REGRESSION
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
results = mae(np.expm1(y_val), np.expm1(lin_reg.predict(X_val)))
print(results)
#MAE = 1274.5598805553673



# DECISION TREE REGRESSOR
max_depth = list(range(8,13,1))
param = {'max_depth': max_depth}
dt_reg = DecisionTreeRegressor( random_state = 20)
dt_cv = GridSearchCV(dt_reg, param, cv = 3, verbose = 1)
dt_cv.fit(X_train,y_train)
print(dt_cv.best_params_)          #'max_depth': 10
dt_cv.best_score_                  #0.43054964105397264



# ELASTIC NET
alpha = list([0.01,0.001])
param_grid = {'alpha' : alpha}
elastic_net = ElasticNet(random_state = 20)
elastic_cv = GridSearchCV(elastic_net, param_grid, verbose = 1, cv = 3)     #Applying gridsearch for lasso regression
elastic_cv.fit(X_train,y_train)

print(elastic_cv.best_params_)    #{'alpha': 0.001}
print(elastic_cv.best_score_)     #0.509733624752

results = mae(np.expm1(y_val), np.expm1(elastic_cv.predict(X_val)))
print(results)  #1272.79920155




# RANDOM FOREST REGRESSOR

rf_reg = RandomForestRegressor(random_state = 20)
rf_reg.get_params()
param_grid = {'n_estimators': [130], 'min_samples_leaf': [15] }

rf_cv = GridSearchCV(rf_reg, param_grid, verbose = 2, cv = 3)
rf_cv.fit(X_train, y_train)
print(rf_cv.best_params_)
rf_cv.best_score_
results = mae(np.expm1(y_val), np.expm1(rf_cv.predict(X_val)))
print(results) #1219.0229862924998





# XGBoost Regressor
from xgboost import XGBRegressor

param = {'n_estimators': [700],'max_depth': [6], 'learning_rate':  [0.1], 'subsample': [0.8]}
xgb_reg = XGBRegressor()
xgb_cv = GridSearchCV(xgb_reg, param, cv = 3, verbose = 3)
xgb_cv.fit(X_train, y_train)
print(xgb_cv.best_params_)
xgb_cv.best_score_

results = mae(np.expm1(y_val), np.expm1(xgb_cv.predict(X_val)))
print(results) 

#1161.5988599517614 --n_estimators = 1000, max_depth=6, learning_rate =  0.1, subsample = 0.8
audio()
slack_message("Execution done", "U9U1CAH2S")


# AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
param = {'n_estimators': [400]}
ab_reg = AdaBoostRegressor(random_state = 20)
ab_cv = GridSearchCV(ab_reg, param, cv = 3, verbose = 3)
ab_cv.fit(X_train, y_train)

print(ab_cv.best_params_)
ab_cv.best_score_
results = mae(np.expm1(y_val), np.expm1(ab_cv.predict(X_val)))
print(results)  # 1624.7352432828143







test_new = complete.iloc[188318:,:]
actual_test = test_new.as_matrix()
predictions = np.expm1(ab_cv.predict(actual_test))

ID = test_data["id"]
predictions = pd.Series(predictions)
out = pd.DataFrame({"id": test_data["id"], 'loss': predictions})
out.to_csv("C:\\Users\\Sanket Keni\\Desktop\\Genesis\\insurance\\out.csv", index=False)



audio()
slack_message("Execution done", "U9U1CAH2S")

















