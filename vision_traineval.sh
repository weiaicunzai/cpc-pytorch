#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
echo "Training the Greedy InfoMax Model on vision data (stl-10)"
python -u -m GreedyInfoMax.vision.main_vision --grayscale --download_dataset --save_dir vision_experiment --model_splits 1 --batch_size 4 --validate

#echo "Testing the Greedy InfoMax Model for image classification"
#python -m GreedyInfoMax.vision.downstream_classification --grayscale --model_path ./logs/vision_experiment --model_num 299
