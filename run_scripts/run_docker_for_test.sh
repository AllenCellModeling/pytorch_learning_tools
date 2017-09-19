nvidia-docker run    \
                    -v $(dirname $(pwd -P)):/root/$(basename $(dirname $(pwd))) \
                    -v /allen/aics/modeling/gregj/results:/root/results \
                    gregj/pytorch_extended \
                    bash -c 'cd pytorch_learning_tools/run_scripts; bash test_script.sh'
