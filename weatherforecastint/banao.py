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
