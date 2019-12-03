## Cloud classification

requirements: torch, torchvision, yaml

The csv files for training and validation is in folder 'csv/'

python3 train.py config/config_train.py: train & validation
python3 test.py config/config_test.py: test & create csv
