# from flask import Flask,request,render_template
# import numpy as np
# import pandas
# import sklearn
# import pickle
# from sklearn.preprocessing import StandardScaler, MinMaxScaler


# app = Flask(__name__)
# # importing model
# model = pickle.load(open('model.pkl','rb'))
# sc = pickle.load(open('standscaler.pkl','rb'))
# ms = pickle.load(open('minmaxscaler.pkl','rb'))

# @app.route('/')
# def index():
#     return render_template("patwal.html")

# @app.route("/predict",methods=['POST'])
# def predict():
#     print(request.form)
#     N = int(request.form.get('Nitrogen',0))
#     P = int(request.form.get('Phosphorus',0))
#     K = int(request.form.get('Potassium',0))
#     Temperature = float(request.form.get('Temperature',0))
#     Humidity = float(request.form.get('Humidity',0))
#     pH = float(request.form.get('pH',0))
#     Rainfall = float(request.form.get('Rainfall',0))

#     list1 = [N, P, K, Temperature, Humidity, pH, Rainfall]
#     single_pred = np.array(list1).reshape(1, -1)

#     scaled_features = ms.transform(single_pred)
#     final_features = sc.transform(scaled_features)
#     prediction = model.predict(final_features   )

#     crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
#                  8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#                  14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#                  19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

#     if prediction[0] in crop_dict:
#         crop = crop_dict[prediction[0]]
#         result = "{} is the best crop to be cultivated right there".format(crop)
#     else:
#         result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
#     return render_template('patwal.html',result = result)


# # python main
# if __name__ == "__main__":
#     app.run(debug=True)




# WORKING CODE=


# import joblib
# from flask import Flask, render_template, request, redirect
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template("HOME.html")

# @app.route('/Predict')
# def prediction():
#     return render_template('Patwal.html')

# @app.route("/form",methods=['POST'])
# def predict():
#     print(request.form)
#     Nitrogen = int(request.form.get('Nitrogen',0))
#     Phosphorus = int(request.form.get('Phosphorus',0))
#     Potassium = int(request.form.get('Potassium',0))
#     Temperature = float(request.form.get('Temperature',0))
#     Humidity = float(request.form.get('Humidity',0))
#     pH = float(request.form.get('pH',0))
#     Rainfall = float(request.form.get('Rainfall',0))

#     values=[Nitrogen,Phosphorus,Potassium,Temperature,Humidity,pH,Rainfall]
    
#     if pH>0 and pH<=14 and Temperature<100 and Humidity>0:
#         joblib.load('banao')
#         model = joblib.load(open('banao','rb'))
#         arr = [values]
#         acc = model.predict(arr)
#         # print(acc)
#         return render_template('result.html', prediction=str(acc))
#     else: 
#         return "Sorry...  Error in entered values in the form Please check the values and fill it again"



# if __name__ == '__main__':
#     app.run(debug=True)



import joblib
import requests
import re
import gtts
import gtts
from flask import Flask, render_template, request, redirect


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("HOME.html")

@app.route('/Predict')
def prediction():
    return render_template('Patwal.html')

@app.route("/form",methods=['POST'])
def predict():
    print(request.form)
    Nitrogen = int(request.form.get('Nitrogen',0))
    Phosphorus = int(request.form.get('Phosphorus',0))
    Potassium = int(request.form.get('Potassium',0))
    Temperature = float(request.form.get('Temperature',0))
    Humidity = float(request.form.get('Humidity',0))
    pH = float(request.form.get('pH',0))
    Rainfall = float(request.form.get('Rainfall',0))

    values=[Nitrogen,Phosphorus,Potassium,Temperature,Humidity,pH,Rainfall]
    
    if pH>0 and pH<=14 and Temperature<100 and Humidity>0:
        joblib.load('banao')
        model = joblib.load(open('banao','rb'))
        arr = [values]
        acc = model.predict(arr)
        # print(acc)
        return render_template('result.html', prediction=str(acc))
    else: 
        return "Sorry...  Error in entered values in the form Please check the values and fill it again"


@app.route('/weather', methods=['POST'])
def weather():
    if request.method == 'POST':
        city = request.form['city']
        weather_data = get_weather(city)
        return render_template('getweather.html', city=city, weather_data=weather_data)
    return render_template('getweather.html', city=None, weather_data=None)

def get_weather(city):
    url = 'https://wttr.in/{}'.format(city)
    result = requests.get(url)
    if result.status_code == 200:
        # Remove ANSI escape codes using regular expression
        weather_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', result.text)
        return weather_data
    
if __name__ == '__main__':
    app.run(debug=True)

