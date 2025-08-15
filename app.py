# from models.xgboost_train import search_best_params

# if __name__=="__main__":
#     search_best_params()
import requests
from flask import Flask, jsonify, request, redirect

app = Flask(__name__)

@app.route('/ping', methods=["GET", "POST"])
def checkin():
    # if request.method=="POST":
    return "Check out!"