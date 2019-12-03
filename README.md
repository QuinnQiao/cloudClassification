## Cloud classification

Requirements: torch(>=0.4.1), torchvision(>=0.2.0), yaml(...)

The csv files for training and validation are in folder 'csv/'.


#### Commands:

python3 train.py config/config_train.py: train & validation

python3 test.py config/config_test.py: test & create csv
