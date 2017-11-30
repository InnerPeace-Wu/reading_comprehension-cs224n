#!/usr/bin/env bash
apt-get -y update
apt-get -y install python-tk
pip install -r requirements.txt

if [ -e "/valohai/inputs/data_squad/data.tar.gz" ]; then
    tar -xvzf /valohai/inputs/data_squad/data.tar.gz
    ls -la ./data/*
    echo "training data in ready"
    time python2 train.py --valohai

else
    time python2 train.py
fi
