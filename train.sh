#!/usr/bin/env bash
sudo apt-get update
sudo apt-get install python-tk

if [ -e "/valohai/inputs/data_squad/data.tar.gz" ]; then
    tar -xvzf /valohai/inputs/data_squad/data.tar.gz
    ls -la ./data/*
    echo "training data in ready"
    time python2 train.py --valohai

else
    time python2 train.py
fi
