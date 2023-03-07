import time
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pre_processing import *
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import pickle
import warnings
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')


data = pd.read_csv("player-value-prediction.csv")
pd.set_option('display.max_rows', None)
X = data.iloc[:, :-1]
#actual y
Y = data["value"]
X,Y=preProcessing(X,Y)

#=======================================================================================================================================
#building model
#Split the data to training and testing sets
#############XGBOOST MODEL
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=0)
#x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, train_size=0.5, shuffle=False)

model = XGBRegressor(n_estimators = 100, learning_rate = 0.2, max_depth = 6)

#Training Time
start=time.time()
model.fit(x_train, y_train)
#validation_data=(x_valid,y_valid)
stop=time.time()
print("training time = ",stop-start," sec.")
#save the trained model   #############
with open('model.pkl', 'wb') as files:
    pickle.dump(model,files)

y_pred = model.predict(x_test)
print('Mean Square Error with XGBOOST', metrics.mean_squared_error(y_test, y_pred))
true_player_value=np.asarray(y_test)[1]
predicted_player_value=y_pred[1]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
print("train accuracy:",model.score(x_train,y_train))
print("Test Accuracy:",r2_score(y_test, y_pred))

############   load the trained model   #############

with open('model.pkl', 'rb') as f:
    load=pickle.load(f)
data=pd.read_csv('player-tas-regression-test.csv')
x = data.iloc[:, :-1]
y=data['value']
x,y=preProcessing(x,y)
y_pred = load.predict(x)
print('Mean Square Error with XGBOOST', metrics.mean_squared_error(y, y_pred))
true_player_value=np.asarray(y)[0]
predicted_player_value=y_pred[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
print("Accuracy:",r2_score(y, y_pred))




