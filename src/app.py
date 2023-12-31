import numpy as np
from flask import Flask, request
import pickle
from flask_cors import CORS,cross_origin

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/',methods=['GET'])
@cross_origin()
def home():
    return "WELCOME TO FLASK APP of SOHAM MEHER"

@app.route("/predict",methods=['POST'])
@cross_origin()
def predict():
    experience = request.json['experience']
    testScore = request.json['testScore']
    interviewScore = request.json['interviewScore']
    final_features = [np.array([experience,testScore,interviewScore])]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return str(output)

if __name__ == "__main__":
    app.run(debug=True)
