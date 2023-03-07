from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


#----------------------------------------------------------Functions--------------------------------------------------------------

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

def date_preprocessing(x):
    if x is np.nan:
        pass
    else:
       x = str(x)
       x = x[-2:]
       return x
#=======================================================================================================================================================

#-------------------------------------------------------Pre-Processing-------------------------------------------------------------


def preProcessing(X,Y):
    # Dealing with uncommon/exceptional cells in body type feature
    # Series.value_counts(normalize=False, sort=True, ascending=False, bins=None(optional, works only with numeric data), dropna=True)
    X.drop(["name", "full_name", "nationality", "birth_date", "club_join_date", "club_team"], axis=1,
           inplace=True)

    lst = []
    count = 0
    for j in X['body_type']:
        if j == "Normal" or j == "Stocky" or j == "Lean":
            lst.append(j)
            count += 1
        else:
            BMI = int(X["weight_kgs"][count]) / (0.01 * int((X["height_cm"][count]) ** 2))
            if BMI >= 25:
                lst.append("Stocky")
                count += 1
            elif BMI <= 18.5:
                lst.append("Lean")
                count += 1
            else:
                lst.append("Normal")
                count += 1
    X['body_type'] = lst



    X = pd.get_dummies(X, columns=['body_type','preferred_foot'], drop_first=True)

    # converts ordinal categorical data into numeric data
    X[['work_rate_attacking', 'work_rate_defense']] = X['work_rate'].str.split('/', expand=True)
    workrate1 = {"Low": 1, "Medium": 2, "High": 3}
    X["work_rate_attacking"] = X["work_rate_attacking"].replace(workrate1)
    workrate2 = {" Low": 1, " Medium": 2, " High": 3}
    X["work_rate_defense"] = X["work_rate_defense"].replace(workrate2)
    X.drop(['work_rate'], axis=1, inplace=True)

    # preprocess contract_end_year column
    X['contract_end_year'] = X['contract_end_year'].apply(date_preprocessing)


    # preprocess position power columns (adding the two values)
    position_power = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',
                      'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']

  
    X[position_power]=X[position_power].apply(lambda x: x.fillna("0+0"))
    

    for col in position_power:
        X[col]=X[col].apply(lambda x:x.split('+'))
        X[col]=X[col].apply(lambda x:int(x[0])+int(x[1])) 
           

    # updating columns with boolean values (0,1)
    #to know if the player is in national team or not
    ll = []
    for i in X['national_rating']:
        if math.isnan(i):
            ll.append(0)
        else:
            ll.append(1)
    X['national_team'] = ll

    X['national_rating'] = X["national_rating"].fillna(0)

    cols = ['traits', 'tags']
    ll = []
    #count number of his values from column
    for i in cols:
        for j in X[i]:
            if j is np.nan:
                cnt = 0
            else:
                j=str(j)
                values = j.split(',')
                cnt = len(values)
            ll.append(cnt)
        X[i] = ll
    

        ll = []
    # converts ordinal categorical data into numeric data
    attaker = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW']
    midline = ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB']
    defender = ['LB', 'LCB', 'CB', 'RCB', 'RB']
    sub = ['SUB', 'RES']
    values_list = []
    for i in X["positions"]:
        value = 0
        lst = []
        lst = i.split(',')
        for j in range(len(lst)):
            if lst[j] in sub:
                value += 1 #least expensive position
            elif lst[j] == 'GK':
                value += 2
            elif lst[j] in defender:
                value += 3
            elif lst[j] in midline:
                value += 4
            elif lst[j] in attaker:
                value += 5   #most expensive potision
        values_list.append(value)
    X["positions"] = values_list

    X["national_team_position"] = X["national_team_position"].fillna("0")
    cols = ["club_position", "national_team_position"]
    value = 0
    for i in cols:
        values_list = []

        for j in X[i]:
            if j in sub:
                value += 1
            elif j == 'GK':
                value += 2
            elif j in defender:
                value += 3
            elif j in midline:
                value += 4
            elif j in attaker:
                value += 5
            else:
                value = 0
            values_list.append(value)
            value = 0
        X[i] = values_list

    # dealing with Nulls
    col_with_nulls = ["wage", "club_rating", "club_jersey_number", "contract_end_year",
                      "release_clause_euro", "national_jersey_number"]
     # Convert nan/null to 0"""
    for col in col_with_nulls:
         X[col]=X[col].fillna(X[col].median())  

    X["contract_end_year"] = X["contract_end_year"].astype(int)
    # scaling
    X_cols = ["id", "wage", "release_clause_euro"]
    X[X_cols] = featureScaling(X[X_cols], 0, 1)
    # Get the correlation between the features
    X_total = X.iloc[:, :]
    X_total["value"] = Y

    ########## Feature Selection ###############
    X.drop(['id', 'height_cm', 'weight_kgs',
             'club_position', 'club_jersey_number', 'national_team_position',
            'national_team',
            'national_jersey_number', 'tags', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',
            'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB',
            'sliding_tackle', 'GK_diving', 'GK_handling', 'GK_kicking',  # -----------shelt el hagat el corr bta3ha 0.0
            'GK_positioning', 'GK_reflexes', 
            'work_rate_defense',  # -----------shelt el hagat el corr bta3ha 0.1
            'positions', 'weak_foot(1-5)', 'heading_accuracy', 'acceleration', 'sprint_speed', 'agility', 'balance',
            'jumping', 'strength', 'aggression',
            'interceptions', 'marking', 'standing_tackle', 'work_rate_attacking', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'LS',
            'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW'], axis=1,
           inplace=True)  # -----------shelt el hagat el corr bta3ha 0.2

    corr = X.corr()
    top_feature = X.iloc[:, :]
    plt.subplots(figsize=(12, 8))
    top_corr = top_feature.corr()
    sns.heatmap(top_corr, annot=True)
    #plt.show()

    X_total['value'] = Y
    # dealing with Nulls in value column
    X_total.dropna(how='any', inplace=True)
    Y = X_total["value"]
    X.drop(['value'], axis=1, inplace=True)
    X_total.to_csv("new fifa.csv", index=False)
    return X,Y


