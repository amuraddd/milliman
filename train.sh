#!/bin/bash

# source <path to the environment if needed>
# cd <working directory>

# search for parameters
# nohup python3 -u -m train --data_path data --data_type train > training_log.log 2>&1 &
# nohup python3 -u -m train --data_path data --data_type train --use_scalar > scalar_training_log.log 2>&1 &

# train
nohup python3 -u -m train --data_path data --data_type train > training_log.log 2>&1 &