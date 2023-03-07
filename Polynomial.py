import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pre_processing import *
from sklearn.metrics import r2_score
import time
import pickle
import warnings
from flask import Flask, render_template, request


warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')

"""data = pd.read_csv("player-value-prediction.csv")
pd.set_option('display.max_rows', None)
X = data.iloc[:, :-1]
#actual y
Y = data["value"]
X,Y=preProcessing(X,Y)
#############################################################
#Building Model

######_____POLYNOMIAL_____MODEL_________
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=False)
poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()

#Training Time
start=time.time()
poly_model.fit(X_train_poly, y_train)
stop=time.time()
print("training time = ",stop-start," sec.")
# predicting on training data-set
# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('Mean Square Error With Polynomial Model', metrics.mean_squared_error(y_test, prediction))
true_player_value=np.asarray(y_test)[0]
predicted_player_value=prediction[0]
print('True value for the player in the test set is : ' + str(true_player_value))
print('Predicted value for the player in the test set is : ' + str(predicted_player_value))
print("Accuracy:",r2_score(y_test, prediction))
with open('poly_features.pkl', 'wb') as files:
    pickle.dump(poly_features,files)
with open('poly_model_save.pkl', 'wb') as files:
    pickle.dump(poly_model,files)"""

########## Load the trained model########################

app = Flask(__name__)

with open('poly_model_save.pkl', 'rb') as f:
    load=pickle.load(f)
with open('poly_features.pkl', 'rb') as f:
    poly_features=pickle.load(f)    

@app.route('/') 
def player() :
    return render_template('player.html')

@app.route('/predict', methods = ['POST', 'GET']) 
def get_prediction() :
    testData = pd.read_csv("player-tas-regression-test.csv")
    pd.set_option('display.max_rows', None)
    x = testData.iloc[:, :-1]
    y = testData["value"]
    if request.method == 'POST' :
        Name = request.form['Name']
        ID = request.form['ID']
        Nationality = request.form['Nationality']
        club_team = request.form['Club Team']
    indx=0
    for i in x['id']:
        if str(i)==str(ID):
            break
        indx+=1
   
    x,y=preProcessing(x,y)
    x_trans=poly_features.transform(x)
    predictions = load.predict(x_trans)
    print('Mean Square Error With Polynomial Model', metrics.mean_squared_error(y, predictions))

    true_player_value=np.asarray(y)[indx]
    predicted_player_value=predictions[indx]
    print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
    print('True value for the first player in the test set in millions is : ' + str(true_player_value))
    print("Accuracy:", r2_score(y, predictions))
    return render_template('prediction.html', value = str(round(predicted_player_value)))
        
   

if __name__ == '__main__' :
    app.run(debug = True)


