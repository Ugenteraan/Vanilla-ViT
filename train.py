'''Training module for the Vision Transformer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from tqdm import tqdm
from loguru import logger

from load_dataset import LoadLabelledDataset
from ViT.ViT import VisionTransformer

DATETIME_NOW = datetime.datetime.now().replace(second=0, microsecond=0) #datetime without seconds & miliseconds.


def train(args):

    #read the config file from args
    with open(args.config, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)
        print("Configuration read successful...Initializing Logger.")
    
    ###Logger init
    if config['logging']['train']['disable_default_loggers']:
        logger.remove(0)

    logging_formatter = config['logging']['train']['formatters'][config['env']] #choose the formatter according to the env.
    
    #add logger to output to file.
    logger.add(f"{config['logging']['train']['log_dir']}{DATETIME_NOW}-{config['logging']['train']['log_filename']}",
                level=config['logging']['train']['level'],
                format=config['logging']['train']['format'],
                backtrace=config['logging']['train']['backtrace'],
                diagnose=config['logging']['train']['diagnose'],
                enqueue=config['logging']['train']['enqueue']
               )

    #add logger to output to console.
    logger.add(sys.stdout,
                level=config['logging']['train']['level'],
                format=config['logging']['train']['format'],
                backtrace=config['logging']['train']['backtrace'],
                colorize=True,
                diagnose=config['logging']['train']['diagnose'],
                enqueue=config['logging']['train']['enqueue']
               )


    ###Extract configurations from the YAML file.

    #Data Configurations
    BATCH_SIZE = config['data']['batch_size']
    IMAGE_SIZE = config['data']['image_size']
    IMAGE_DEPTH = config['data']['image_depth']
    DATASET_FOLDER = config['data']['dataset_folder']
    NUM_WORKERS = config['data']['num_workers']
    SHUFFLE = config['data']['shuffle']
    USE_RANDOM_HORIZONTAL_FLIP = config['data']['use_random_horizontal_flip']
    RANDOM_AFFINE_DEGREES = config['data']['random_affine']['degrees']
    RANDOM_AFFINE_TRANSLATE = config['data']['random_affine']['translate']
    RANDOM_AFFINE_SCALE = config['data']['random_affine']['scale']
    COLOR_JITTER_BRIGHTNESS = config['data']['color_jitter']['brightness']
    COLOR_JITTER_HUE = config['data']['color_jitter']['hue']

    #Model configurations
    MODEL_SAVE_FOLDER = config['model']['model_save_folder']
    MODEL_NAME = config['model']['model_name']
    MODEL_SAVE_FREQ = config['model']['model_save_freq']
    N_SAVED_MODEL_TO_KEEP = config['model']['N_saved_model_to_keep']
    TRANSFORMER_DEPTH = config['model']['transformer_depth']
    FEEDFORWARD_PROJECTION_DIM = config['model']['feedforward_projection_dim']
    NUM_HEADS = config['model']['num_heads']
    PROJECTION_KEYS_DIM = config['model']['projection_keys_dim']
    PROJECTION_VALUES_DIM = config['model']['projection_values_dim']
    MLP_RATIO = config['model']['mlp_ratio']
    ATTN_DROPOUT_PROB = config['model']['attn_dropout_prob']
    MLP_DROPOUT_PROB = config['model']['mlp_dropout_prob']

    #Training configurations
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and config['training']['device'] == 'gpu' else 'cpu')
    LOAD_CHECKPOINT = config['training']['load_checkpoint']
    LOAD_CHECKPOINT_EPOCH = config['training']['load_checkpoint_epoch']
    END_EPOCH = config['training']['end_epoch']
    START_EPOCH = config['training']['start_epoch']
    COSINE_UPPER_BOUND_LR = config['training']['cosine_upper_bound_lr']
    COSINE_LOWER_BOUND_LR = config['training']['cosine_lower_bound_lr']
    WARMUP_START_LR = config['training']['warmup_start_lr']
    WARMUP_STEPS = config['training']['warmup_steps']
    NUM_EPOCH_TO_RESTART_LR = config['training']['num_epoch_to_restart_lr']
    COSINE_UPPER_BOUND_WD = config['training']['cosine_upper_bound_wd']
    COSINE_LOWER_BOUND_WD = config['training']['cosine_lower_bound_wd']
    USE_BFLOAT16 = config['training']['use_bfloat16']
    USE_NEPTUNE = config['training']['use_neptune']
    

    if USE_NEPTUNE:
        import neptune

        NEPTUNE_RUN.init_run(project=cred.NEPTUNE_PROJECT,
                             api_token=cred.NEPTUNE_API_TOKEN)
        
        #we have partially unsupported types of data in config. Hence this method.
        NEPTUNE_RUN['parameters'] = neptune.utils.stringify_unsupported(config)

        
    logger.info("Initializing Vision Transformer...")

    VISION_TRANSFORMER = VisionTransformer(image_size=IMAGE_SIZE, 
                                           patch_size=PATCH_SIZE, 
                                           patch_embedding_dim=PATCH_EMBEDDING_DIM, 
                                           image_depth=IMAGE_DEPTH, 
                                           device=DEVICE, 
                                           num_classes=NUM_CLASSES, 
                                           transformer_network_depth=TRANSFORMER_DEPTH,
                                           num_heads=NUM_HEADS,
                                           mlp_ratio=MLP_RATIO,
                                           projection_keys_dim=PROJECTION_KEYS_DIM,
                                           projection_values_dim=PROJECTION_VALUES_DIM,
                                           attn_dropout_prob=ATTN_DROPOUT_PROB,
                                           mlp_dropout_prob=MLP_DROPOUT_PROB
                                           )

    transforms_compose_list = [transforms.ColorJitter(brightness=COLOR_JITTER_BRIGHTNESS, hue=COLOR_JITTER_HUE),
                          transforms.RandomAffine(degrees=RANDOM_AFFINE_DEGREES, translate=RANDOM_AFFINE_TRANSLATE, scale=RANDOM_AFFINE_SCALE),
                          transforms.ToTensor()
                          ]
    #insert the random horizontal flip to the list at the beginning if it's true.
    if USE_RANDOM_HORIZONTAL_FLIP:
        transforms_compose_list.insert(0, transforms.RandomHorizontalFlip())
    #insert the lambda function to convert grayscale images (with depth 1) to RGB (sort of) images. This is required since some images in dataset might be originally grayscale.
    if IMAGE_DEPTH == 3:
        #this process should be AFTER the image has been converted to tensor.
        transforms_compose_list.append(transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)))



    DATASET_MODULE = LoadLabelledDataset(dataset_folder_path=DATASET_FOLDER,
                                         image_size=IMAGE_SIZE,
                                         image_depth=IMAGE_DEPTH,
                                         train=True,
                                         transform=transforms_compose_list,
                                         logger=logger)
                                        )
        
        

