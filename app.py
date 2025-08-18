"""
Flask based API to trigger training and inference through routes.
Lots more work to be done to serve this, but will make it more robust.
"""
import requests
import subprocess
from flask import Flask, jsonify, request, redirect

app = Flask(__name__)

@app.route('/checkin', methods=["GET", "POST"])
def checkin():
    """
    Health check route./
    Reurns:
    - (str): check out!
    """
    return "Check out!"

@app.route('/train', methods=["GET", "POST"])
def train():
    """
    Route to trigger training.
    """
    subprocess.run(
        ["bash", "train.sh"]
    )
    return "Training started.."

@app.route('/predict', methods=["GET", "POST"])
def predict():
    """
    Route to trigger prediction generation.
    """
    subprocess.run(
        ["bash", "predict.sh"]
    )
    return "Making and saving predictions.."

if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        port=8000
    )
