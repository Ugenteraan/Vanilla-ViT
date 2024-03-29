'''Training module for the Vision Transformer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from tqdm import tqdm
from loguru import logger


DATETIME_NOW = datetime.datetime.now().replace(second=0, microsecond=0) #datetime without seconds & miliseconds.


def train(args):

    #read the config file from args
    with open(args.config, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)
        print("Configuration read successful...Initializing Logger.")
    
    ###Logger init
    if config['disable_default_loggers']:
        logger.remove(0)

    logging_formatter = config['logging']['formatters'][config['env']] #choose the formatter according to the env.
    
    #add logger to output to file.
    logger.add(f"{config['log_dir']}{DATETIME_NOW}-{config['log_filename']}",
                level=config['logging']['level'],
                format=config['logging']['format'],
                backtrace=config['logging']['backtrace'],
                diagnose=config['logging']['diagnose'],
                enqueue=config['logging']['enqueue']
               )

    #add logger to output to console.
    logger.add(sys.stdout,
                level=config['logging']['level'],
                format=config['logging']['format'],
                backtrace=config['logging']['backtrace'],
                colorize=True,
                diagnose=config['logging']['diagnose'],
                enqueue=config['logging']['enqueue']
               )


    ###Extract configurations from the YAML file.

    #Data Configurations
    BATCH_SIZE = config['data']['batch_size']
    IMAGE_SIZE = config['data']['image_size']
    IMAGE_DEPTH = config['data']['image_depth']
    DATASET_FOLDER = config['data']['dataset_folder']



        
        

