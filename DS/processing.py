from torch.utils.data import Dataset

class PreDataLoader(Dataset):
    def __init__(self, data, candle_count=50, start=200, stop=450, normalization_pred=True, vers=1, lbl=1, candle_in_patch=1):
        """Create DataLoader for your DataSet
            :Parameters:
                tickers : str, list
                    List of tickers to download
                data: DataFrame
                    DataFrame of candle parameters
                pred_size: int
                    Prediction size(Count of columns with prediction parameters)
                label_size: int
                    Label size(Count of columns with label parameters)
                butch_size: int
                    Butch size for NN
                start: int
                    Number of row that you want to start
                stop: int
                    Number of row that you want to finish
            """
        self.data = data.copy()
        self.candle_count = candle_count
        self.start = start
        self.stop = stop
        self.normalization_pred = normalization_pred
        self.predictions = []
        self.labels = []
        self.batches = []
        self.version = vers
        self.lbl = lbl
        self.candle_in_patch = candle_in_patch
        self.create_exit_data()


    def create_exit_data(self):
        '''Not availaible now'''

        prediction = []
        label = []

        for i in range(self.start+self.candle_count, self.stop):
            prediction = []
            label = []
            high = []
            low = []
            ema = []
            assets = []
            candles = []

            for j in range(self.candle_count):
                high.append(self.data.High[i-self.candle_count+j])
                low.append(self.data.Low[i - self.candle_count + j])
                ema.append(self.data.EMA[i - self.candle_count + j])
                if self.version == 1:
                    assets.append(self.data.Assets[i-1])

            if self.version == 2:
                assets.append(self.data.Assets[i-1])

            if (self.normalization_pred == True):
                max0 = max(high)
                max1 = max(ema)
                rmax = max(max0, max1)
                min0 = min(low)
                min1 = min(ema)
                rmin = min(min0, min1)

                for itr in range(self.candle_count):
                    high[itr] = (high[itr]-rmin)/(rmax-rmin)
                    low[itr] = (low[itr]-rmin)/(rmax-rmin)
                    ema[itr] = (ema[itr]-rmin)/(rmax-rmin)

            # Сюда вписать кодину бьющую окно по свечам

            for candle_iter in range(self.candle_count):
                candle = []
                candle.append(high[candle_iter])
                candle.append(low[candle_iter])
                candle.append(ema[candle_iter])
                candle.append(assets[candle_iter])
                candles.append(candle)

            # TODO: Тут если что дописать код для битья на патчи по несколько свечей

            # prediction.append(high)
            # prediction.append(low)
            # prediction.append(ema)
            # prediction.append(assets)
            # prediction.append(candles)

            if self.lbl == 1:
                label.append(self.data.Take_profit[i - 1])
                label.append(self.data.Stop_loss[i - 1])
            elif self.lbl == 2:
                label.append(self.data.Take_profit[i - 1])
            elif self.lbl == 3:
                label.append(self.data.Stop_loss[i - 1])

            self.predictions.append(candles)
            self.labels.append(label)

        self.batches.append(self.predictions)
        self.batches.append(self.labels)

        # print(self.predictions)
        # print('#'*30)
        # print(self.labels)


