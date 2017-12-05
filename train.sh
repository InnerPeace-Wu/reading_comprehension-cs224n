#!/usr/bin/env bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    #statements
    key="$1"

    case $key in
        -lr|--learning_rate)
        LR=$2
        shift
        shift
        ;;
        -kp|--keep_prob)
        KEEP_PROB=$2
        shift
        shift
        ;;
        -bs|--batch_size)
        BATCH_SIZE=$2
        shift
        shift
        ;;
        -es|--embed_size)
        EMBED_SIZE=$2
        shift
        shift
        ;;
        -ss|--state_size)
        STATE_SIZE=$2
        shift
        shift
        ;;
        -reg|--reg)
        REG=$2
        shift
        shift
        ;;
        *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
# len=${#array[@]}
# EXTRA_ARGS=${array[@]:6:$len}
# EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

if [ -e "/valohai/inputs/data_squad/data.tar.gz" ]; then
    apt-get -y update
    apt-get -y install python-tk
    pip install -r requirements.txt
    tar -xvzf /valohai/inputs/data_squad/data.tar.gz
    ls -la ./data/*
    echo "training data in ready"
    time python2 train.py --lr ${LR} \
        --keep_prob ${KEEP_PROB} \
        --batch_size ${BATCH_SIZE} \
        --embed_size ${EMBED_SIZE} \
        --state_size ${STATE_SIZE} \
        --reg ${REG} \
        --valohai \
        # ${EXTRA_ARGS}
else
    time python2 train.py --lr ${LR} \
        --keep_prob ${KEEP_PROB} \
        --batch_size ${BATCH_SIZE} \
        --embed_size ${EMBED_SIZE} \
        --state_size ${STATE_SIZE} \
        --reg ${REG} \
        # --test \
        # ${EXTRA_ARGS}

fi


