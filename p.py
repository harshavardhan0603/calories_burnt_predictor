import numpy as np
import pandas as pd
from xgboost import XGBRegressor

calories = pd.read_csv("C:\projects\calories_burnt_predictor\predictor\data_sets\calories.csv")
exercise_data = pd.read_csv('C:\projects\calories_burnt_predictor\predictor\data_sets\exercise.csv')
calories_data = pd.concat([exercise_data,calories["Calories"]], axis= 1)
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

    # """Separating features and targets"""

X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

    # """Model Training"""

model = XGBRegressor()
model.fit(X, Y)

    # """Predictive system"""

input_data = (1,27,154,58,10,81,39.8)

    # changing input_data to a dataframe
n = np.asarray(input_data)
input_data_set = {'Gender' : [n[0]], 'Age': [n[1]], 'Height': [n[2]], 'Weight': [n[3]], 'Duration': [n[4]], 'Heart_Rate': [n[5]], 'Body_Temp' : [n[6]]} 
df = pd.DataFrame(input_data_set)
prediction = model.predict (df)   
print(prediction[0])