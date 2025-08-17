"""
Flask based API to trigger training and inference through routes.
Lots more work to be done to serve this, but will make it more robust.
"""
import requests
from flask import Flask, jsonify, request, redirect

app = Flask(__name__)

@app.route('/ping', methods=["GET", "POST"])
def checkin():
    return "Check out!"