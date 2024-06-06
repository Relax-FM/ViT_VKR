"""

    

"""

import yaml
import torch

from DS.load import load_data
from DS.processing import PreDataLoader
from DS.labels import get_label
from DS.Dataset import DataSet


class Dataloader():
    def __init__(self):
        options_path = 'config.yml'
        with open(options_path, 'r') as options_stream:
            options = yaml.safe_load(options_stream)
        self.dataset_options = options.get('dataset')
        data_path = self.dataset_options.get('data_file_name')
        EMA_N = self.dataset_options.get('EMA')
        self.data = load_data(data_path, EMA_N).data  # Получили все строчки из экселя
        # Прилетают они в формате серий (всегда можно узнать дату и значение)

    def get_dataloader(self, start: int | None = None, stop: int | None = None, test: bool = False) -> torch.utils.data.DataLoader:
        """

        Метод создает даталоадер с нужными настройками.\n
        С нужной начальной позиции и до нужной конечной позиции.

        :param start: С какой позиции
        :param stop: По какую позицию
        :param test: Валидация (Да/Нет)
        :return: loader: Возвращает даталоадер
        """

        loader_options = self.dataset_options.get('train_loader')
        if test is True:
            loader_options = self.dataset_options.get('test_loader')

        if start is None:
            start = loader_options.get('start')
        if stop is None:
            stop = loader_options.get('stop')

        pdl = PreDataLoader(self.data, candle_count=self.dataset_options.get('candle_count'),
                            start=start, stop=stop,
                            normalization_pred=self.dataset_options.get('normalization'),
                            vers=self.dataset_options.get('vers'),
                            lbl=get_label(self.dataset_options.get('label'), False if test else True))  # Получаем два списка
        # В первом списке - входные данные, во втором - label данные
        # High, Low, EMA, Assets [2, [200, [4, 50]]]          2 - predict label      200 - размер ДС      4 - кол-во параметров у свечи              50 - свечей
        ds = DataSet(pdl.batches)

        loader = torch.utils.data.DataLoader(
            ds, shuffle=loader_options.get('shuffle'),
            batch_size=loader_options.get('batch_size'), num_workers=loader_options.get('num_workers'),
            drop_last=loader_options.get('drop_last')
        )
        return loader
