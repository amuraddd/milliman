#!/bin/bash

# make predictions
nohup python3 -u -m predict > predict_log.log 2>&1 &