import time
import pandas as pd
from sklearn.model_selection import train_test_split
from pre_processing import *
import pickle
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')

"""data = pd.read_csv("player-classification.csv")
pd.set_option('display.max_rows', None)
X = data.iloc[:, :-1]
# actual y
Y = data["PlayerLevel"]
X, Y = preProcessing(data, X, Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=100, learning_rate=0.2)
start_F = time.time()
ABC.fit(X_train, y_train)
stop_F = time.time()
traintime = stop_F - start_F
print("Training time = ", traintime, " sec.")
start_P = time.time()
y_prediction = ABC.predict(X_test)
stop_P = time.time()
test_time = stop_P - start_P
print("Testing time = ", test_time, " sec.")
accuracy = np.mean(y_prediction == y_test) * 100
print("The achieved accuracy using Adaboost is " + str(accuracy))
true_player_value = np.asarray(y_test)[2872]
predicted_player_value = y_prediction[2872]
print('True value for the  player in the test set is : ' + str(true_player_value))
print('Predicted value for the  player in the test set is : ' + str(predicted_player_value))
with open('Adaboost_model.pkl', 'wb') as files:
    pickle.dump(ABC, files)"""

#################### loading file ###################################

app = Flask(__name__)
with open('Adaboost_model.pkl', 'rb') as f:
    load = pickle.load(f)


@app.route('/')
def playerclass():
    return render_template('playerclass.html')
@app.route('/classify', methods = ['POST', 'GET']) 
def get_classification() :    
    data = pd.read_csv('player-tas-classification-test.csv')
    x = data.iloc[:, :-1]
    y = data['PlayerLevel']
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
    x, y = preProcessing(data, x, y)
    y_pred = load.predict(x)
    accuracy = np.mean(y_pred == y) * 100
    print("The achieved accuracy using Adaboost is " + str(accuracy))
    true_player_value = np.asarray(y)[indx]
    predicted_player_value = y_pred[indx]
    print('True value for the player in the test set is : ' +
          str(true_player_value))
    print('Predicted value for the player in the test set is : ' +
          str(predicted_player_value))
      
    return render_template('classification.html', value = predicted_player_value)    
if __name__ == '__main__' :
    app.run(debug = True)

