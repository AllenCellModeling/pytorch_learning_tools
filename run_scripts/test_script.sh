cd ../

code_dir=$PWD"/pytorch_learning_tools"

export PYTHONPATH=$code_dir:$PYTHONPATH
export PATH=$code_dir:$PATH

train_model.py \
    --gpu_ids [2] \
    --batch_size 50 \
    --nepochs 50 \
    --lr 5E-4 \
    --model_name conv3D \
    --data_save_path ./data/dp_h5.pyt \
    --save_dir ./output/classifier3D/ \
    --train_module train_single_target \
    --data_dir /root/results/ipp_dataset_cellnuc_seg_curated_8_24_17/ \
    --save_state_iter 1 \
    --save_progress_iter 1

