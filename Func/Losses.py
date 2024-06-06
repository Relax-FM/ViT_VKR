import torch.nn as nn

def get_losser(option_losser: dict):
    name = option_losser.get('name', None)
    if name is None:
        raise NotImplementedError(
            'Loss func is None. Please, add to config file')
    name = name.lower()

    losser = None
    if name in ('mseloss','mse'):
        losser = nn.MSELoss()
        print('MSE loss')
    elif name in ('l1loss','l1'):
        losser = nn.L1Loss()
        print('L1 loss')
    elif name in ('cel', 'crossentropy'):
        losser = nn.CrossEntropyLoss()
        print('CrossEntropy loss')
    else:
        raise NotImplementedError(
            f'Loss func [{name}] is not recognized. losses.py doesn\'t know {[name]}')

    return losser