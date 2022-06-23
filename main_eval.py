#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

#
from torch_ema import ExponentialMovingAverage
import numpy as np

# Utils
from tqdm import tqdm

#
import models
from data.dataset.ImageLabelDataSet import ImageLabelDataSet
from Configure import Configure

###################################################################################

# GLOBAL SETTINGS (the filename to store or load from training result folder)
LOADING_MODEL_PATH = f'checkpoint_ema_model.pth'
LOADING_EMA_PATH = F'checkpoint_ema.pth'
PREDICTED_FILE_PATH = f'prediction.csv'

# 
def main_eval(config : Configure, logdir):
    '''
    load weights form .saved/@logdir folder, and then use the model to evaluate the testing dataset

    @ Params:
        1. config (Configure): configure file
        2. logdir (str): a training folder that stores model & eam weights
    
    @ Returns: None (the testing result will automatically saved in the logdir folder)
    '''

    # Step 1 : Get the mapping functions that maps index to class label
    # Use training dataset to know the relationship between predicted index & class label (folder name)
    # will get a dict that maps index to class label
    ds = ImageLabelDataSet(config.dataset.root_train, transform_set=None)
    idx_to_class = {}
    for k in ds.class_to_idx:
        idx_to_class[ds.class_to_idx[k]] = k

    # Step 2 : Model Define & Load
    model = getattr(models, config.model.model_name)(config)  
    device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
    # loading model weights from assigned location (GLOBAL SETTINGS)
    model = model.to(device)
    if torch.cuda.is_available() is True:
        model = model.cuda()
    model.load_state_dict(torch.load(f'./saved/{logdir}/{LOADING_MODEL_PATH}', map_location=device))

    # loading ema weights from assigned location (GLOBAL SETTINGS)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema.load_state_dict(torch.load(f'./saved/{logdir}/{LOADING_EMA_PATH}', map_location=device))

    # Step 3 : Testing DataSet & DataLoader
    resize = resize = (config.model.input_size, config.model.input_size)
    # the most simple transforms, which involves only resize and normalize
    transform_set = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds_test = ImageLabelDataSet(config.dataset.root_eval, transform_set=transform_set)
    test_loader = DataLoader(ds_test, batch_size=config.dataset.eval_batch_size, shuffle=False, num_workers=config.dataset.num_workers)

    # Step 4 : Make prediction via trained model
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []
    with ema.average_parameters():

        # Iterate the validation set by batches.
        for batch in tqdm(test_loader):

            # A batch consists of image data and corresponding labels.
            imgs, _ = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))
            predictions += logits.argmax(dim=-1)

    # Step 5 : Save predictions into the file.
    with open(f'./saved/{logdir}/{PREDICTED_FILE_PATH}', "w") as f:

        # The first row must be "Id, Category"
        f.write("filename,category\n")

        #
        imgs_file_names = get_fileName(ds_test)

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in enumerate(predictions):
            ans = idx_to_class[pred.item()]
            imgs_file_name = imgs_file_names[i].split("\\")
            f.write(f"{imgs_file_name[-1]},{ans}\n")

    # 
    # # Step 6 : Explanation & Visualization
    # # get_confidence_score(model, loader=test_loader, use_gpu_index=config.use_gpu_index, batch_size=config.batch_size, outpu_file_path=f'{output_file_path[:-3]}_Confidence.csv')
    # get_confidence_score(model, loader=test_loader, use_gpu_index=config.use_gpu_index, batch_size=BATCH_SIZE, outpu_file_path=f'{output_file_path[:-3]}_Confidence.csv')

def get_fileName(ds):
    '''
    Utils function for test(), will return a list of image filenames of @ds

    @ Params:
        1. ds (torch.data.dataset):
            a instance if pytorch dataset

    @ Retruns:
        1. fileNames (List[str]):
            a list that includes the file names of all testing images
    '''

    # 
    fileNames = []
    for i in range(len(ds.imgs)):
        fileNames.append(ds.imgs[i][0])

    return fileNames

###################################################################################