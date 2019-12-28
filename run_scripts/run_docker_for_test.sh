#!/usr/bin/env bash

PAR_DIR=$(dirname $(pwd -P))

nvidia-docker run \
    -v ${PAR_DIR}:/root/$(basename ${PAR_DIR}) \
    -v /allen/aics/modeling/gregj/results:/root/results \
    ${USER}/pytorch_extended:master \
    bash -c 'cd pytorch_learning_tools/run_scripts; bash test_script.sh'
