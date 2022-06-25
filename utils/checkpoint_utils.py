import torch 

# GLOBAL SETTINGS
CHECKPOINT_SAVE_PATH = f'checkpoint.pth'

def save_checkpoint(epoch : int, 
                    model : torch.nn.Module,  
                    ema, 
                    optimizer : torch.optim.Optimizer,
                    logdir : str):
    '''
    @ Params:
    @ Returns:
    '''
    
    # 
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(), 
        'optim_state_dict': optimizer.state_dict()
    }
    # 
    torch.save(checkpoint, f'./saved/{logdir}/{CHECKPOINT_SAVE_PATH}')


def load_checkpoint(model : torch.nn.Module, 
                    ema, 
                    optimizer : torch.optim.Optimizer, 
                    logdir : str):
    '''
    @ Params:
    @ Returns:
    '''
    
    # load in the checkpoint (as a format of dictionary)
    checkpoint = torch.load(f'./saved/{logdir}/{CHECKPOINT_SAVE_PATH}')

    # load the weight into the model instances 
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    return epoch, model, ema, optimizer