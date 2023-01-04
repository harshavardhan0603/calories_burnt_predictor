from django.shortcuts import render
import pandas as pd
from xgboost import XGBRegressor
# Create your views here.

def home(request):

    

    calories = pd.read_csv("C:\projects\calories_burnt_predictor\predictor\data_sets\calories.csv")
    exercise_data = pd.read_csv('C:\projects\calories_burnt_predictor\predictor\data_sets\exercise.csv')
    calories_data = pd.concat([exercise_data,calories["Calories"]], axis= 1)
    calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)

    """Separating features and targets"""

    X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
    Y = calories_data['Calories']

    """Model Training"""

    model = XGBRegressor()
    model.fit(X, Y)

    """Predictive system"""
    if request.method == "POST" :

        input_data_set = {'Gender' : [int(request.POST["gender"])], 'Age': [int(request.POST["age"])], 'Height': [float(request.POST["height"])], 'Weight': [float(request.POST["weight"])], 'Duration': [float(request.POST["duration"])], 'Heart_Rate': [int(request.POST["hr"])], 'Body_Temp' : [float(request.POST["temp"])]} 
        df = pd.DataFrame(input_data_set)
        prediction = model.predict (df)

        return render(request,"home.html" , {"calories":prediction[0]} )

    return render(request,"home.html"  )
