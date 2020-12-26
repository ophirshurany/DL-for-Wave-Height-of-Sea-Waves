from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import os.path as osp


class ModelForSequence:
    """
    Class for inheritance for all models that support accepting the data as a sequence
    """
    def __init__(self, name, look_back, data_dim, build_config_description=''):
        """
        :param name: string representing model name
        :param look_back: how many samples back each prediction will be fed
        :param data_dim: how many different features (dimensions) will be fed into model
        :param build_config_description_string: String representation of the whole configuration
        """
        self.name = name
        self.data_dim = data_dim
        self.look_back = look_back
        self.conf_str = build_config_description
        # implemented by each model separately
        self.model = None
        self.model_save_dir = osp.join('output', 'models')

    def fit(self, X_train, y_train, val_data, num_epochs, batch_size):
        """
        :param X_train: training data
        :param y_train: training labels
        The following are relevant only for the neural networks for which
        these parameters will be used for training and for early stopping
        :param val_data: tuple with X_val, y_val
        :param num_epochs: epochs for model training
        :param batch_size: batch size for training
        :return: trained model
        """
        raise NotImplementedError("Model classes should implement this")

    def predict(self, X):
        """
        Calculates predictions for input data X
        :param X: data
        :return: prediction results
        """
        raise NotImplementedError("Model classes should implement this")

    def flatten_steps_back(self, multi_dim_data):
        """
        For use in models that expect to receive single samples and not sequential data
        Takes a sequence of dimensions (n, m ,d)
        where n is number of samples, m is steps back and d is number of features
        and flattening it into an array with n samples, each of size m*d
        :return:
        """
        input_dim = multi_dim_data.shape[2]
        assert(input_dim == self.data_dim)
        steps_back = multi_dim_data.shape[1]
        assert(steps_back == self.look_back)
        return multi_dim_data.reshape(-1, input_dim*steps_back)


class LSTMModel(ModelForSequence):
    """
    LSTM model class
    two current structures available - with one and two hidden layers
    """
    def __init__(self, *, look_back, input_dimension, build_config_description='', num_lstm_layers=1):
        super().__init__("LSTM", look_back, input_dimension, build_config_description)
        self.model = self.build(look_back, input_dimension, num_lstm_layers)
        self.str_repr = 'lstm{}'.format(num_lstm_layers)
        self.history = None

    def build_one_layer_lstm(self, look_back, data_dimension):
        """
        :return: model with one hidden LSTM layer
        """
        model = Sequential()
        model.add(LSTM(100, input_shape=(look_back, data_dimension), return_sequences=False,
                       activation='relu'))
        model.add(Dense(1, name="output_l"))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def build_two_layer_lstm(self, look_back, data_dimension):
        """
        :return: model with two hidden LSTM layers
        """
        model = Sequential()
        model.add(LSTM(80, input_shape=(look_back, data_dimension), return_sequences=True, activation='relu'))
        model.add(LSTM(10, activation='relu'))
        model.add(Dense(1, name="output_l"))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def build(self, look_back, data_dimension, num_lstm_layers):
        """
        :param look_back: num steps back for prediction from sequence
        :param data_dimension: number of features
        :param num_lstm_layers: which format of lstm (with two hidden layers or one)
        :return: built model
        """
        if 1 == num_lstm_layers:
            return self.build_one_layer_lstm(look_back, data_dimension)
        else:
            return self.build_two_layer_lstm(look_back, data_dimension)

    def fit(self, X_train, y_train, val_data=None, num_epochs=10, batch_size=10):
        """
        :param X_train: training data
        :param y_train: training labels
        :param val_data: tuple with X_val, y_val
        :param num_epochs: epochs for model training
        :param batch_size: batch size for training
        :return: trained model
        """
        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        saved_model_path = osp.join(self.model_save_dir, '{}.h5'.format(self.conf_str))
        # make checkpoint for saving model during training
        mc = ModelCheckpoint(saved_model_path,
                             monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        self.history = self.model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
                                      verbose=2, validation_data=val_data, callbacks=[es, mc],
                                      shuffle=False)
        # load saved model - if had early stopping this is relevant
        self.model = load_model(saved_model_path)
        return self.model

    def predict(self, X):
        """
        Gives prediction from input sequences shape (n, m, d) where n is number of samples,
        m is sequence length, and d is feature dimensions - number of different features
        :param X: input data for prediction. in sequence format.
        :return: predictions
        """
        return self.model.predict(X, batch_size=1)


class FCNNModel(ModelForSequence):
    """
    Fully connected neural network (two hidden layers, 32 neurons, 16 neurons and regression output
    supporting data received in sequence format
    """
    def __init__(self, *, look_back, input_dimension, build_config_description):
        super().__init__("FCNN", look_back, input_dimension,
                         build_config_description)
        self.model = self.build(look_back*input_dimension)
        self.history = None

    def build(self, input_size):
        """
        Build fully connected layer with two hidden layer and output neuron for the regression task
        :param input_size:
        :return: model
        """
        model = Sequential()
        model.add(Dense(32, input_dim=input_size, name='dense_first_hidden'))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(1))
        # try with and without sigmoid
        model.add(Activation('sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def fit(self, X_train, y_train, val_data=None, num_epochs=10, batch_size=10):
        """
        :param X_train: training data
        :param y_train: training labels
        :param val_data: tuple with X_val, y_val
        :param num_epochs: epochs for model training
        :param batch_size: batch size for training
        :return: trained model
        """
        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        saved_model_path = osp.join(self.model_save_dir, '{}.h5'.format(self.conf_str))
        # make checkpoint for saving model during training
        mc = ModelCheckpoint(saved_model_path,
                             monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # for fully connected we want to flatten the series
        X_train_fc = self.flatten_steps_back(X_train)
        X_val_fc = self.flatten_steps_back(val_data[0])
        val_data_fc = (X_val_fc, val_data[1])
        self.history = self.model.fit(X_train_fc, y_train, epochs=num_epochs, batch_size=batch_size,
                                      verbose=2, validation_data=val_data_fc, callbacks=[es, mc],
                                      shuffle=False)
        # since we have model checkpoint saving the one with the minimal loss, best to reload this one,
        # is likely to be best version. Especially if training was not optimal and/or if model diverged/
        self.model = load_model(saved_model_path)
        return self.model

    def predict(self, X):
        """
        Gives prediction from input sequences shape (n, m, d) where n is number of samples,
        m is sequence length, and d is feature dimensions - number of different features
        :param X: input data for prediction. in sequence format.
        :return: predictions
        """
        X_fc = self.flatten_steps_back(X)
        return self.model.predict(X_fc, batch_size=1)


class XGBoostModel(ModelForSequence):
    """
    XGBoost - tree based gradient boosting model, supporting data received in sequence format
    """
    def __init__(self, *, look_back, input_dimension, build_config_description):
        super().__init__("XGBoost", look_back, input_dimension,
                         build_config_description)
        # currently with default parameters, can be changed of course
        self.model = XGBRegressor()

    def fit(self, X_train, y_train, val_data=None, num_epochs=None, batch_size=None):
        """
        :param X_train: training data
        :param y_train: training labels
        following params are here only for interface compatibility, although not relevant here
        :param val_data:
        :param num_epochs:
        :param batch_size:
        :return: trained model
        """
        X_train_flat = self.flatten_steps_back(X_train)
        self.model.fit(X_train_flat, y_train)
        return self.model

    def predict(self, X):
        """
        Gives prediction from input sequences shape (n, m, d) where n is number of samples,
        m is sequence length, and d is feature dimensions - number of different features
        :param X: input data for prediction. in sequence format.
        :return: predictions
        """
        X_f = self.flatten_steps_back(X)
        return self.model.predict(X_f)


class RandomForestModel(ModelForSequence):
    # Random Forest - tree based bagging model, supporting data received in sequence format
    def __init__(self, *, look_back, input_dimension, build_config_description):
        super().__init__("RF", look_back, input_dimension,
                         build_config_description)
        # currently with default parameters, can be changed of course
        self.model = RandomForestRegressor()

    def fit(self, X_train, y_train, val_data=None, num_epochs=None, batch_size=None):
        """
        :param X_train: training data
        :param y_train: training labels
        following params are here only for interface compatibility, although not relevant here
        :param val_data:
        :param num_epochs:
        :param batch_size:
        :return: trained model
        """
        X_train_flat = self.flatten_steps_back(X_train)
        self.model.fit(X_train_flat, y_train)
        return self.model

    def predict(self, X):
        """
        Gives prediction from input sequences shape (n, m, d) where n is number of samples,
        m is sequence length, and d is feature dimensions - number of different features
        :param X: input data for prediction. in sequence format.
        :return: predictions
        """
        X_f = self.flatten_steps_back(X)
        preds = self.model.predict(X_f)
        return preds.reshape(-1, 1)
