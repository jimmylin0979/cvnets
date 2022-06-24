#
from cProfile import label
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

# Other 3rd party library
import numpy as np
from torch_ema import ExponentialMovingAverage
from ptflops import get_model_complexity_info
from sklearn.model_selection import train_test_split

# Utils 
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

# Local package
import models
from data.dataset.ImageLabelDataSet import ImageLabelDataSet
from optim.scheduler import GradualWarmupScheduler
import utils.mix as mix
from Configure import Configure

###################################################################################

# Global Setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main_train(config : Configure, logdir : str):

    '''
    Construct corresponding model and training environment via @config, 
    and the whole training stats & saving weights will store in ./saved/@logdir folder 

    @ Params:
        1. config (Configure): 
        2. logdir (str): 
    
    @ Returns: None
        
    '''

    # Step 1 : prepare logging writer
    writer = SummaryWriter(log_dir=f'./saved/{logdir}')

    # Step 2 : 
    logging.info(config.model.model_name)
    model = getattr(models, config.model.model_name)(config)
    # TODO : Implement auto reseum
    # if config.load_model:
    #     model.load_state_dict(torch.load(config.model_path))
    
    # 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 
    # Metrics : FLOPs, Params
    # resize = (224, 224) if config.model_name != 'ConvNeXt' else (384, 384)
    resize = (config.model.input_size, config.model.input_size)
    macs, params = get_model_complexity_info(model, (3, resize[0], resize[1]), as_strings=True, print_per_layer_stat=False, verbose=False)
    logging.info('ptflops : ')
    logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # Step 3 : DataSets
    # Data Augumentation
    transform_set = [
        transforms.RandomResizedCrop((resize[0])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # transforms.RandAugment()
    ]
    transform_set = transforms.Compose([

        # # Reorder transform randomly
        transforms.RandomOrder(transform_set),
        # Resize the image into a fixed shape
        transforms.Resize(resize),
        # 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 
        transforms.RandomErasing()
    ])
    ds = ImageLabelDataSet(config.dataset.root_train, transform_set=transform_set)

    # Step 
    # Deal with imbalance dataset : 
    #   For the classification task, we use cross-entropy as the measurement of performance.
    #   Since the wafer dataset is serverly imbalance, we add class weight to make it classifier better
    class_weights = [1 - (ds.targets.count(c))/len(ds) for c in range(config.model.n_classes)]
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.scheduler.lr, weight_decay=1e-8)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    # TODO : Implement Auto resume
    # if config.load_model:
    #     ema = ema.load_state_dict(torch.load(config.ema_path))

    # scheduler_warmup is chained with schduler_steplr
    scheduler_steplr = CosineAnnealingLR(optimizer, T_max=config.scheduler.tmax)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=config.scheduler.warmup_epoch, after_scheduler=scheduler_steplr)

    # Step 4
    # train_loader, valid_loader = get_loader(ds)
    ds_train, ds_valid = get_train_valid_ds(config, ds)

    # Step 5
    history = {'train_acc' : [], 'train_loss' : [], 'valid_acc' : [], 'valid_loss' : []}
    best_epoch, best_epoch_ema, best_loss, best_acc_ema, best_acc = 0, 0, 1e100, 0, 0
    nonImprove_epochs = 0

    # GradientWarmScheduler : 
    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()

    # 
    for epoch in range(config.scheduler.max_epoch):

        # update the learning rate before each epoch
        scheduler_warmup.step(epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        logging.info('=' * 80)
        logging.info(f'Epoch {epoch}, LR = {optimizer.param_groups[0]["lr"]}')

        # 
        train_loader = DataLoader(ds_train, batch_size=config.dataset.train_batch_size, shuffle=True, num_workers=config.dataset.num_workers, pin_memory=True)
        valid_loader = DataLoader(ds_valid, batch_size=config.dataset.train_batch_size, shuffle=False, num_workers=config.dataset.num_workers, pin_memory=True)

        # 
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, ema, config)
        logging.info(f"[ Train | {epoch + 1:03d}/{config.scheduler.max_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        valid_acc, valid_loss = valid(model, valid_loader, criterion, None)
        logging.info(f"[ Valid | {epoch + 1:03d}/{config.scheduler.max_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        valid_acc_ema, valid_loss_ema = valid(model, valid_loader, criterion, ema)
        logging.info(f"[ Valid | {epoch + 1:03d}/{config.scheduler.max_epoch:03d} ] loss = {valid_loss_ema:.5f}, acc = {valid_acc_ema:.5f} (EMA)")

        # Tensorboard Visualization
        writer.add_scalar("Train/train_acc", train_acc, epoch)
        writer.add_scalar("Valid/valid_acc", valid_acc, epoch)
        writer.add_scalar("Valid/valid_acc_ema", valid_acc_ema, epoch)
        writer.add_scalar("Train/train_loss", train_loss, epoch)
        writer.add_scalar("Valid/valid_loss", valid_loss, epoch)
        writer.add_scalar("Valid/valid_loss_ema", valid_loss_ema, epoch)

        # Save best model depending on top-1 accuracy
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), f'./saved/{logdir}/checkpoint_best.pth')
            logging.info(f'Saving model with acc {valid_acc:.4f} and loss {valid_loss:.4f}')

        # EarlyStop
        # if the model improves, save a checkpoint at this epoch
        if valid_acc_ema > best_acc_ema:
            best_loss = valid_loss_ema
            best_acc_ema = valid_acc_ema
            best_epoch = epoch
            torch.save(model.state_dict(), f'./saved/{logdir}/checkpoint_ema_model.pth')
            torch.save(ema.state_dict(), f'./saved/{logdir}/checkpoint_ema.pth')
            logging.info(f'Saving model with acc {valid_acc_ema:.4f} and loss {valid_loss_ema:.4f} (EMA)')
            nonImprove_epochs = 0
        else:
            nonImprove_epochs += 1

        # Stop training if your model stops improving for "config['early_stop']" epochs.    
        if nonImprove_epochs >= config.scheduler.earlystop_epoch:
            break
    
    # 
    logging.info(f'Best epoch: {best_epoch} with acc {best_acc:.4f}')
    logging.info(f'Best epoch: {best_epoch_ema} with acc {best_acc_ema:.4f} (EMA)')

    # 
    writer.flush()
    writer.close()

    # # Step 6 : Explanation & Visualization
    # get_confidence_score(model, loader=valid_loader, use_gpu_index=config.use_gpu_index, batch_size=config.batch_size, outpu_file_path=f'{logdir}/last-prediction-Confidence.csv')

###################################################################################

def get_train_valid_ds(config : Configure, ds):

    '''
    Split the @ds into train and valid dataset, the split ratio will be determine by @config

    @ Params:
        1. config
        2. ds
    
    @ Returns:
        1. ds_train: train dataset
        2. ds_valid: valid dataset
    '''

    # Split the train/test with each class should appear on both train/test dataset
    valid_split = config.dataset.train_valid_split

    indices = list(range(len(ds)))  # indices of the dataset
    train_indices, valid_indices = train_test_split(indices, test_size=valid_split, stratify=ds.targets, random_state=42)
    
    # Creating sub dataset from valid indices
    # Do not shuffle valid dataset, let the image in order
    valid_indices.sort()
    ds_valid = torch.utils.data.Subset(ds, valid_indices)

    ds_train = torch.utils.data.Subset(ds, train_indices)

    return ds_train, ds_valid

###################################################################################

def train(model, train_loader, criterion, optimizer, ema, config):
    
    '''
    @ Params:
        1. model:
        2. valid_loader:
        3. criterion:
        4. optimizer:
        5. ema: whether to use ema to perform the valid process  
    
    @ Returns:
        1. np.mean(accs) (float): the train accuracy in current epoch 
        2. np.mean(losses) (float): the train loss in current epoch
    '''

    # Make sure the model is in train mode before training.
    model.train()

    # Training history for current epoch
    losses, accs = [], []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.to(device)

        # 
        # Now supports cutmix, mixup 2 mix-based augmetations
        do_mix = True if np.random.rand(1) < config.mix.prob else False
        mix_method = 'cutmix' if np.random.rand(1) > 0.5 else 'mixup'
        #
        target_a, target_b, lam = None, None, None
        if do_mix:
            # preparation for cutmix
            if mix_method == 'cutmix':
                
                # generate mixed sample
                lam = np.random.beta(config.mix.cutmix_beta, config.mix.cutmix_beta)
                if torch.cuda.is_available():
                    rand_index = torch.randperm(imgs.size()[0]).cuda()
                else:
                    rand_index = torch.randperm(imgs.size()[0]).cpu()        
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = mix.rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
            
            # preparation for mixup
            elif mix_method == 'mixup':
                labels = labels.to(device)
                imgs, targets_a, targets_b, lam = mix.mixup_data(imgs, labels, alpha=0.2, use_cuda=torch.cuda.is_available())
                imgs, targets_a, targets_b = map(Variable, (imgs, targets_a, targets_b))
        
        else:
            labels = labels.to(device)

        if do_mix:
            logging.debug(f'current mixed method : {mix_method}')

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = None 
        #
        if do_mix:
            #
            if mix_method == 'mixup':
                loss = mix.mixup_criterion(criterion, logits, targets_a, targets_b, lam)

            #
            elif mix_method == 'cutmix':
                target_a = target_a.to(device)
                target_b = target_b.to(device)
                loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
        else:
            loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Update the parameters with computed gradients.
        optimizer.step()
        ema.update()

        # # Clip the gradient norms for stable training.
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Compute the training stats for current batch, and then append to the history
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        accs.append(acc.item())
        losses.append(loss.item())

    return np.mean(accs), np.mean(losses)

def valid(model, valid_loader, criterion, ema=None):

    '''
    @ Params:
        1. model:
        2. valid_loader:
        3. criterion:
        4. ema: whether to use ema to perform the valid process  
    
    @ Returns:
        1. np.mean(accs) (float): the valid accuracy in current run 
        2. np.mean(losses) (float): the valid loss in current run
    '''
    
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # training stats in current epoch
    accs, losses = [], []

    if ema is not None:

        with ema.average_parameters():

            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = model(imgs.to(device))

                # We can still compute the loss (but not the gradient).
                losses.append(criterion(logits, labels.to(device)).item())

                # Compute the accuracy for current batch.
                accs.append((logits.argmax(dim=-1) == labels.to(device)).float().mean().item())
    
    else:
        
        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            losses.append(criterion(logits, labels.to(device)).item())

            # Compute the accuracy for current batch.
            accs.append((logits.argmax(dim=-1) == labels.to(device)).float().mean().item())

    return np.mean(accs), np.mean(losses)

###################################################################################