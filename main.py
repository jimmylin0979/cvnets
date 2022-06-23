#
import argparse
import os
import shutil
import yaml

# Configure
from Configure import Configure

#
from main_train import main_train
from main_eval import main_eval

###################################################################################

if __name__ == '__main__':

    # 1.
    parser = argparse.ArgumentParser(description='AICUP - Orchid Classifier')
    parser.add_argument('--mode', '-m', default='train', type=str, required=True,
                                help='Selecting whether to train or to evaluate')
    parser.add_argument('--logdir', default='model', type=str, required=True,
                                help='The folder to store the training stats of current model')
    args = parser.parse_args()

    # 2
    # 2.1. read yaml file and parse into dictionary
    config = None
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    # 2.2. then parse the dictionary to make it able to get attribute via dot (.)
    #       ex. config.dataset.name
    config = Configure(config)

    # 3.
    # mode train
    if args.mode == 'train':
        main_train(config, args.logdir)
    # mode eval
    elif args.mode == 'eval':
        main_eval(config, args.logdir)

###################################################################################