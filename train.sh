#!/bin/bash

source /home/azm0269@auburn.edu/venv/bin/activate
cd /home/azm0269@auburn.edu/milliman

nohup python3 -u -m train --data_path data --data_type train > training_log.log 2>&1 &
nohup python3 -u -m train --data_path data --data_type train --use_scalar > scalar_training_log.log 2>&1 &
# tail -f training_log.log