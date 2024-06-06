"""

    2) Нужно сделать функцию создания нового датасета
    5) Заимпортить мой кастомный датасет и лоадер.
    7) Сделать вывод в Excel по обучению и каждому этапу дообучения
    8) Оформить это всё как класс и вынести общие переменные
    9) Можно прям в этом файле сделать помимо обучения еще и тесты сразу и выводить метрики только по тестам в целом

"""

import numpy as np
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast
import torch

import time
import datetime

from DS.get_dataloader import Dataloader
from Func.Functions import *
from Func.testing_version import *


def feedforward(epochs: int, train_loader, device: str, optimizer, use_amp: bool, CNNet, loss_fn,
                scaler, epsilon: float, batch_size: int, lbl_from_conf: str, flag = False):
    """

    Функция выполняет роль прямого прогона обучения НС.\n
    Учитываются указанные настройки и параметры.\n
    Выводятся нужные метрики.\n

    :param epochs: Кол-во эпох для обучения
    :param train_loader: Даталоадер для обучения
    :param device: На каком устройстве обучать ('cuda'/'cpu')
    :param optimizer: Функция оптимизации (adam/sgd)
    :param use_amp: Использовать сжатие вещественных типов данных (True/False)
    :param CNNet: Модель сверточной нейронной сети
    :param loss_fn: Функция потерь (CEL/L1/MSE)
    :param scaler: Эт штука чет с градиентами делает
    :param epsilon: Эпсилон для метрики точности ответов НС
    :param batch_size: Размер бача (Изначально 25)
    :param lbl_from_conf: Указание о том какой параметр предугадывается(stop-loss/take-profit)
    :param flag: Делать вывод метрик обучения или нет
    :return: Имя файла, где храниться модель
    """
    CNNet = CNNet.to(device)
    max_acc = -1.0

    labels = []
    results = []
    accuracies = []
    losses = []

    print('#' * 40)
    print('#' * 40)
    print('\nНачал обучение CNN')
    start_time = time.time()

    for epoch in range(epochs):
        loss_val = 0
        acc_val = 0
        # acc_val_profit = 0
        # acc_val_loss = 0
        labels.clear()
        results.clear()
        for sample in train_loader:  # (pbar := tqdm(train_loader))
            info, lbl = sample['info'], sample['label']
            info = info.to(device)
            lbl = lbl.to(device)
            optimizer.zero_grad()

            with autocast(use_amp):
                pred = CNNet(info)
                loss = loss_fn(pred, lbl)
            scaler.scale(loss).backward()
            loss_item = loss.item()
            loss_val += loss_item

            scaler.step(optimizer)
            scaler.update()

            acc_current = accuracy_v3(pred.cpu().float(), lbl.cpu().float(), epsilon=epsilon)
            acc_val += acc_current

            labels.append(lbl.cpu())
            results.append(pred.cpu())

        # print(f"Epoch : {epoch + 1}")  # TODO: Раскоментить
        # print(f"Loss : {loss_val / len(train_loader)}")  # TODO: Раскоментить
        losses.append(loss_val / len(train_loader))
        # print(f"Acc : {acc_val / (len(train_loader) * batch_size)}")  # TODO: Раскоментить
        accuracies.append(acc_val / (len(train_loader) * batch_size))
        if acc_val / (len(train_loader) * batch_size) > max_acc:
            max_acc = acc_val / (len(train_loader) * batch_size)
        # print(f"Acc profit : {acc_val_profit / (len(train_loader)*batch_size)}")
        # print(f"Acc loss : {acc_val_loss / (len(train_loader)*batch_size)}")
        # print(f"len : {len(train_loader)*batch_size}")
    if flag:
        print(f"Epochs : {str(epochs)}")
        print(f"Max accuracy : {max_acc}")
        print(f"Min loss: {min(losses)}")
    print(f'Full time learning : {time.time() - start_time}')

    labels = TenArrToNP(labels)
    results = TenArrToNP(results)
    # print(labels)
    if flag:
        avg_lbl, avg_res = average(labels, results)
        standard_deviation = calculated_standard_deviation(labels, results)
        error = calculated_error(standard_deviation, avg_lbl)
        max_error = calculated_max_error(labels, results)

    curDate = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    # print(curDate)  # TODO: Раскоменьтить можно
    file_name = curDate + "_" + lbl_from_conf + "_" + str(epochs) + "_" + device.upper() + ".pth"  # TODO: Раскоменьтить
    # file_name = lbl_from_conf + "_" + device.upper() + ".pth"  # TODO: Закоменьтить
    # print("\'" + file_name + "\'")  # TODO: Раскоменьтить можно
    path_name = "models/no_additional/" + file_name
    torch.save(CNNet.state_dict(), path_name)

    h = np.linspace(1, len(losses), len(losses))

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    ax.plot(h[:], losses[:])
    ax.set_title("Loss for epoch.")
    ax.set_xlabel("Axis epoch")
    ax.set_ylabel("Axis loss")
    ax.grid()
    plt.savefig(f'results/photo/no_additional/losses/{file_name.replace(".pth",".png")}')
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    ax.plot(h, accuracies)
    ax.set_title("accuracy for epoch.")
    ax.set_xlabel("Axis epoch")
    ax.set_ylabel("Axis accuracy")
    ax.grid()
    plt.savefig(f'results/photo/no_additional/accuracy/{file_name.replace(".pth", ".png")}')
    # plt.show()

    # Здесь уже тест

    print("\nНачало тестирования")
    feedforward_test(CNNet, loss_fn, epsilon, epochs, file_name)

    return "\'" + file_name + "\'"


def additional_learning(start: int, stop: int, step: int, epochs: int, device: str, optimizer, use_amp: bool, CNNet,
                        loss_fn, scaler, epsilon: float, batch_size: int, lbl_from_conf: str, flag=False):
    """

    Функция выполняет роль дообучения НС.\n
    Учитываются указанные настройки и параметры.\n
    Выводятся нужные метрики.\n

    :param start: Начальная позиция даталоадера
    :param stop: Конечная позиция даталоадера
    :param step: Шаг изменения позиции даталоадера
    :param epochs: Количество эпох (можно сделать списком, чтоб потом можно было настраивать каждый этап)
    :param device: На каком устройстве запускать ('cuda', 'cpu')
    :param optimizer: Функция оптимизации (adam/sgd)
    :param use_amp: Использовать сжатие вещественных типов данных (True/False)
    :param CNNet: Модель сверточной нейронной сети
    :param loss_fn: Функция потерь (CEL/L1/MSE)
    :param scaler: Эт штука чет с градиентами делает
    :param epsilon: Эпсилон для метрики точности ответов НС
    :param batch_size: Размер бача (Изначально 25)
    :param lbl_from_conf: Указание о том какой параметр предугадывается(stop-loss/take-profit)
    :param flag: Делать вывод метрик обучения или нет
    :return: Имя файла, где храниться модель
    """
    CNNet = CNNet.to(device)
    dataloader = Dataloader()

    test_loader = dataloader.get_dataloader(test=True)

    print('#' * 40)
    print('#' * 40)
    print('\nНачал дообучение CNN')
    start_time = time.time()

    # Сюда пихать переменные которые отвечают за прогон всех эпох всех даталоадеров всех бачей
    count_step = 0
    predict_draw_list = []
    label_draw_list = []
    test_all_rel = []
    test_loss_val = 0
    test_acc_val = 0
    flag = False

    # TODO: ВАЖНО сделать здесь тестовый даталодер и сделать для него тестовое покрытие

    for (current_step, test_batch) in zip(range(start, stop, step), test_loader):
        count_step += 1

        # Тут тестирование
        with torch.no_grad():
            test_info, test_lbl = test_batch['info'], test_batch['label']
            test_info = test_info.to(device)
            test_lbl = test_lbl.to(device)

            test_pred = CNNet(test_info)

            test_loss = loss_fn(test_pred, test_lbl)
            test_loss_item = test_loss.item()
            test_loss_val += test_loss_item

            test_acc_current = accuracy_v3(test_pred.cpu(), test_lbl.cpu(), epsilon=epsilon)
            test_acc_val += test_acc_current

            test_relation = (test_pred[0] / test_lbl[0]) * 100
            test_all_rel.append(abs(test_relation.detach().cpu().numpy()[0]))
        predict_draw_list.append(test_pred.detach().cpu().numpy()[0][0])
        label_draw_list.append(test_lbl.detach().cpu().numpy()[0][0])



        # Сюда пихать переменные которые отвечают за прогон всех эпох одного даталоадера всех бачей
        current_loader = dataloader.get_dataloader(start=current_step-250, stop=current_step, additional=True)

        for epoch in range(epochs):

            # Сюда пихать переменные которые отвечают за прогон одной эпохи одного даталоадера всех бачей

            for sample in current_loader:  # (pbar := tqdm(train_loader))
                info, lbl = sample['info'], sample['label']
                info = info.to(device)
                lbl = lbl.to(device)
                optimizer.zero_grad()

                with autocast(use_amp):
                    pred = CNNet(info)
                    loss = loss_fn(pred, lbl)

                scaler.scale(loss).backward()
                loss_item = loss.item()
                # loss_val += loss_item

                scaler.step(optimizer)
                scaler.update()

                # acc_current = accuracy_v3(pred.cpu().float(), lbl.cpu().float(), epsilon=epsilon)
                # acc_val += acc_current

                # labels.append(lbl.cpu())
                # results.append(pred.cpu())
    if flag:
        print(f"Steps : {str(count_step)}")
        print(f"Epochs : {str(epochs)}")
    print(f'Full time learning : {time.time() - start_time}\n')

    curDate = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    # print(curDate)  # TODO: Раскоменьтить можно
    file_name = curDate + "_" + lbl_from_conf + "_" + str(epochs) + "_" + device.upper() + ".pth"  # TODO: Раскоменьтить
    # file_name = lbl_from_conf + "_" + device.upper() + ".pth"  # TODO: Закоменьтить
    # print("\'" + file_name + "\'")  # TODO: Раскоменьтить можно
    path_name = "models/additional/" + file_name
    torch.save(CNNet.state_dict(), path_name)

    print("\nНачало тестирования")

    print(f'Loss of all DS: {test_loss_val / len(test_loader)}')
    print(f'Acc of all DS: {test_acc_val / len(test_loader)}')
    print(f'Length of DS: {len(test_loader)}')
    print(f'Count of right answer : {test_acc_val}')
    print(f"AV_rel: {(sum(test_all_rel) / len(test_all_rel))}")

    avg_lbl, avg_res = average(label_draw_list, predict_draw_list)
    standard_deviation = calculated_standard_deviation(label_draw_list, predict_draw_list)
    error = calculated_error(standard_deviation, avg_lbl)
    max_error = calculated_max_error(label_draw_list, predict_draw_list)

    length_list = [i for i in range(len(test_loader))]

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))
    ax.plot(length_list, predict_draw_list, label="predict", color='blue')
    ax.plot(length_list, label_draw_list, label="label", color='red')
    ax.set_xlabel(f'$N_i$')
    ax.set_ylabel('Stop_loss')

    ax.set_title(f'Тест - {epochs} эпох', fontsize=20)
    ax.legend()
    ax.grid(True)
    plt.savefig(f'results/photo/additional/test/{file_name.replace(".pth", ".png")}')

    return "\'" + file_name + "\'"