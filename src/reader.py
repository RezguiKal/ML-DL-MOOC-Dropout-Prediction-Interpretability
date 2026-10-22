from os.path import join
import config.conf
import pandas as pd

#############################################################
class DataReader:

    """
        Initialises the DataReader object
        Params:
            - filename (str): the name of the dataset to read
            - lag (int): how many chunks of the timeseries to 
                         consider before prediction

        return object (DataReader)
    """
    def __init__(self, filename, lag=7):
        filename = filename + '.csv' if '.csv' not in filename\
            else filename

        self.filename = filename
        self.lag = lag
#############################################################
#############################################################
    """
        Reads the data

        return tuple of X and y where X represents the
            timeseries (np.ndarray)
            and y is the ground truth (np.array)
    """
    def read_data(self):
        timeseries = pd.read_csv(self.filename)

        persons_col = timeseries.columns[0]

        truth_col = timeseries.columns[-1]

        timeseries_len = len(timeseries)\
            // len(timeseries[persons_col].unique())

        y = timeseries[truth_col][0::timeseries_len].values

        timeseries = timeseries.drop(columns=[persons_col,\
            truth_col])

        X = timeseries.values.reshape(timeseries.shape[0]\
            // timeseries_len,\
            timeseries_len, timeseries.shape[-1])

        X = X[:,0:self.lag,:]

        print(f'X shape {X.shape}\ty shape {y.shape}')

        return X, y
#############################################################
#############################################################
