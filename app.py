import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
# đọc model
app = Flask(__name__)
model = joblib.load(open('randomforestmodel.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = int(prediction[0])
    ans = ''
    if output == 0:
        ans = 'Kecimen'
    elif output == 1:
        ans = 'Besni'
    return render_template('index.html', prediction_text='Kết quả dự đoán loại nho là : {}'.format(ans))
if __name__ == "__main__":
    app.run(debug=True)