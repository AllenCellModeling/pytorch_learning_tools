cd ..
python train_model.py --gpu_ids 2 --batch_size 50 --nepochs 50 --lr 5E-4 --model_name conv3D --data_save_path ./data/dp_h5.pyt --save_dir ./output/classifier3D/ --train_module classifier_train --data_dir /root/results/ipp_dataset_cellnuc_seg_curated_8_24_17/ --save_state_iter 1 --save_progress_iter 1 
