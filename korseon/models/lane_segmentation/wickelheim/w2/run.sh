# run using: !sh ./run.sh
gdown --id 1LTdUXzUWcnHuEEAiMoG42oAGuJggPQs8 -O ./data/culane/driver_100_30frame.tar.gz
tar -xvzf ./data/culane/driver_100_30frame.tar.gz -C ./data/culane/
python ./train.py --data_dir data/culane --dataset culane --learning_rate 5e-4