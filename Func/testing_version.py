import numpy as np
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast
import torch

import time
import datetime

from DS.get_dataloader import Dataloader
from Func.Functions import *


def feedforward_test(model, loss_fn, epsilon, train_epochs, file_name):
    model.cpu()
    test_loader = Dataloader().get_dataloader(test=True)
    prediction_list = []
    label_list = []
    all_rel = []
    length_list = [i for i in range(len(test_loader))]

    loss_val = 0
    acc_val = 0

    for sample in test_loader:
        with torch.no_grad():
            info, label = sample['info'], sample['label']

            pred = model(info)

            loss = loss_fn(pred, label)
            loss_item = loss.item()
            loss_val += loss_item

            acc_current = accuracy_v3(pred, label, epsilon=epsilon)
            acc_val += acc_current
            relation = (pred[0] / label[0]) * 100
            # print(relation)
            all_rel.append(abs(relation.detach().numpy()[0]))
        prediction_list.append(pred.detach().numpy()[0][0])
        label_list.append(label.detach().numpy()[0][0])
        # print(f'pred: {str(pred.detach().numpy()[0][0])}\tlbl: {str(label.detach().numpy()[0][0])}')  # TODO: Раскоментить
        # print(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
    print(f'Loss of all DS: {loss_val / len(test_loader)}')
    print(f'Acc of all DS: {acc_val / len(test_loader)}')
    print(f'Length of DS: {len(test_loader)}')
    print(f'Count of right answer : {acc_val}')
    print(f"AV_rel: {(sum(all_rel) / len(all_rel))}")

    avg_lbl, avg_res = average(label_list, prediction_list)
    standard_deviation = calculated_standard_deviation(label_list, prediction_list)
    error = calculated_error(standard_deviation, avg_lbl)
    max_error = calculated_max_error(label_list, prediction_list)

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    ax.plot(length_list, prediction_list, label="predict", color='blue')
    ax.plot(length_list, label_list, label="label", color='red')
    ax.set_xlabel(f'$N_i$')
    ax.set_ylabel('Stop_loss')

    ax.set_title(f'Тест - {train_epochs} эпох', fontsize=20)
    ax.legend()
    ax.grid(True)
    plt.savefig(f'results/photo/test/{file_name.replace(".pth", ".png")}')
    # plt.show()
