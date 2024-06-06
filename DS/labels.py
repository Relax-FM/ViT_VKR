from enum import Enum

class Labels(Enum):
    take_profit = ['take_profit', 'tp', 't_p', '2']
    stop_loss = ['stop_loss', 'sl', 's_l', '3']
    together = ['all', 'tg', '1']

def get_label(param, print_flag = True):
    name = param
    if name is None:
        raise NotImplementedError(
            'Label name is None. Please, add to config file')
    name = name.lower()

    label = None
    if name in Labels.together.value :
        label = 1
        if print_flag:
            print('together version')

    elif name in Labels.take_profit.value :
        label = 2
        if print_flag:
            print('take_profit version')

    elif name in Labels.stop_loss.value :
        label = 3
        if print_flag:
            print('stop_loss version')
    else:
        raise NotImplementedError(
            f'Label [{name}] is not recognized. labels.py doesn\'t know {[name]}')

    return label