import pandas as pd
from torch.utils.data import Dataset


class load_data():
    def __init__(self, path: str, N: int):
        self.file_path = path
        self.N = N
        self.data = self.dataload_xlsx()
        self.data = self.create_EMA()


    def dataload_xlsx(self):
        '''
        Use for loading data about candles from file
        :arg:
        file - name of file to loading data (file.xls)
        :return: pd.DateFrame about candle in format (Date, Open, High, Low, Close, EMA200, Assets, Take_profit, Stop_loss)
        '''
        data = pd.read_excel(f'DS/Excel/{self.file_path}.xlsx')
        data = data.set_index('Date')
        del data['Open']
        del data['EMA200']
        # print(data)
        return data


    def create_EMA(self):
        data = self.data
        # data[f'EMA{self.N}'] = data['Close'].ewm(span=self.N, adjust=False).mean() # Считаем нужный ЕМА
        data.insert(loc=3, column=f'EMA', value=data['Close'].ewm(span=self.N, adjust=False).mean())
        del data['Close']
        # print(data)
        return data
