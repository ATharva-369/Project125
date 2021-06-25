from flask import Flask, jsonify, request
from Classifier import getPrediction
app = Flask(__name__)
@app.route('/predict-alphabet',methods=["POST"])

def predict_data():
    image = request.files.get("alphabet")
    print(image)
    prediction = getPrediction(image)
    return jsonify ({
        "prediction":prediction
    },200)

if __name__ == "__main__":
    app.run(debug=True)    