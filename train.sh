#!/bin/bash

source /home/azm0269@auburn.edu/venv/bin/activate
cd /home/azm0269@auburn.edu/milliman

python3 -m train > training_log.log 2>&1 
wait

deactivate