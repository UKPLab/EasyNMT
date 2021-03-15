#!/bin/bash

# turn on bash's job control
set -m

# Download the needed model
python -c "from easynmt import EasyNMT; import os; EasyNMT(os.getenv('EASYNMT_MODEL', 'opus-mt'))"

# Start the primary process and put it in the background
/start_backend.sh &

# Start the helper process
/start_frontend.sh &
  
# the my_helper_process might need to know how to wait on the
# primary process to start before it does its work and returns
  
  
# now we bring the primary process back into the foreground
# and leave it there
# fg %1

# Naive check if both processes are still running
while sleep 60; do
  ps aux |grep gunicorn_conf_frontend.py |grep -q -v grep
  PROCESS_1_STATUS=$?
  ps aux |grep gunicorn_conf_backend.py |grep -q -v grep
  PROCESS_2_STATUS=$?


  if [ $PROCESS_1_STATUS -ne 0 -o $PROCESS_2_STATUS -ne 0 ]; then
    echo "One of the processes has already exited."
    exit 1
  fi
done