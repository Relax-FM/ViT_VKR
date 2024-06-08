"""

    В этом файле происходит обучение нейронной сети.
    Чтобы запустить необходимо просто настроить config.yml под нужные параметры.
    Также нужно настроить конфигурации интерпретатора. Указать ему путь до этого файла.

"""
import random

import torch
import yaml

from DS.get_dataloader import Dataloader
from Func.NN import ViT
from Func.Losses import get_losser
from Func.Optimizer import get_optimizer
from Func.learning_version import *

options_path = 'config.yml'
with open(options_path, 'r') as options_stream:
    options = yaml.safe_load(options_stream)

network_options = options.get('network')
dataset_options = options.get('dataset')
train_loader_options = dataset_options.get('train_loader')

accuracu_opt = network_options.get('accuracy')
epsilon = accuracu_opt.get('epsilon')

start = train_loader_options.get('start')
stop = train_loader_options.get('stop')
batch_size = train_loader_options.get('batch_size')  # TODO: Глянуть может можно убрать
lbl_from_conf = dataset_options.get('label')  # TODO: Глянуть может можно убрать
flag_for_print = dataset_options.get('flag')


train_loader = Dataloader().get_dataloader()

transformer_options = network_options.get('transformer')

ViTNet = ViT(window_size=dataset_options.get('candle_count'),
             patch_size=50,
             num_transformer_layers=transformer_options.get('n_enc_layer'),
             embedding_dim=transformer_options.get('n_embd'),
             mlp_size=transformer_options.get('mlp_size'),
             num_heads=transformer_options.get('n_head'),
             attn_dropout=transformer_options.get('att_drop'),
             mlp_dropout=transformer_options.get('resid_drop'),
             embedding_dropout=transformer_options.get('embd_drop'),
             num_classes=transformer_options.get('n_classes'))

loss_fn = get_losser(network_options.get('loss'))
optimizer = get_optimizer(ViTNet.parameters(), network_options.get('optimizer'))

device = network_options.get('device')
use_amp = network_options.get('use_amp')
scaler = torch.cuda.amp.GradScaler()
epochs = train_loader_options.get('epochs')

ViTNet = ViTNet.to(device)
loss_fn = loss_fn.to(device)

# Все зарандомили.
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Обычное обучение НС
train = feedforward(epochs, train_loader, device, optimizer, use_amp, ViTNet, loss_fn, scaler, epsilon, batch_size, lbl_from_conf, flag_for_print)


print("#"*30)
print("#"*30)
print(f"\nNo additional : {train}")
