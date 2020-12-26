import configparser
import Models


class TestInstanceParams:
    def __init__(self, input_data_str_repr, model_str_repr, desc_str_addition='',
                 target_col="b_hs", is_target_in_input=True, pred_forward_hrs=4,
                 look_back_hrs=12, time_sample_res_minutes=10, model_class=Models.LSTMModel,
                 model_args={"num_epochs": 5, "batch_size": 50},
                 k=5, num_folds_to_run=None, config_file_path="run_conf.ini"):
        """"
        Class for holding all relevant params for training and evaluating test scenario
        Built for ease of use for running and comparing multiple scenarios
        :param input_data_str_repr: description string of run will have the format:
        <pred_target><forward hours>h_<time_res_min>m<input_data>_lb<look_back hours>h_<model_name><optional addition>
        this parameter is the one used for the <input_data> tag.
        :param model_str_repr: string representing the name of model
        :param desc_str_addition: optional suffix for desc str
        :param target_col: prediction target
        :param is_target_in_input: True if same data source as the one forecasting for is included in training data
        :param pred_forward_hrs: The time delta forward for which the forecast will be predicted for
        :param look_back_hrs: How many hours of data (backwards) is used for the prediction
        :param time_sample_res_minutes: sampling resolution. currently data is every 10 minutes. should be
        a whole multiplier of 10 (equal to some int * 10)
        :param model_class: model class used for building model
        :param model_args: args for model
        :param k: number of folds to split data to
        :param num_folds_to_run: optional - how many folds for cross-validation. if not given, will run on all folds
        num_folds_to_run must be <= to k. if smaller than k, folds used will be the last ones (those that the test
        data is closer to the end of the time window)
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        self.target_col = target_col
        self.is_target_in_input = is_target_in_input
        self.pred_forward_hrs = pred_forward_hrs
        self.look_back_hrs = look_back_hrs

        self.model_class = model_class
        self.model_args = model_args
        self.k = k
        if not num_folds_to_run:
            self.num_folds_to_run = k
        else:
            self.num_folds_to_run = num_folds_to_run
        self.time_res_min = time_sample_res_minutes
        self.desc_str = self.build_desc_str(input_data_str_repr, model_str_repr, desc_str_addition)
        # column name of physical model
        self.phys_col = self.find_phys_col()
        self.samples_in_hr = 60 / time_sample_res_minutes
        self.downsample_ratio = time_sample_res_minutes / 10
        assert (self.samples_in_hr == int(self.samples_in_hr) and
                self.downsample_ratio == int(self.downsample_ratio))
        self.samples_in_hr = int(self.samples_in_hr)
        self.downsample_ratio = int(self.downsample_ratio)
        self.pred_forward = self.pred_forward_hrs * self.samples_in_hr
        self.train_steps = self.look_back_hrs * self.samples_in_hr

    def build_desc_str(self, input_data_repr, model_str_repr, addition):
        """"
        :param input_data_repr: string representing the sources used for model input
        :param model_str_repr: string representing the structure of the chosen model
        :param addition: any suffix if such addition is desired to representation
        :return: the string itself. the format of the result is:
         <pred_target><forward hours>h_<time_res_min>m<input_data>_lb<look_back hours>h_<model_name><optional addition>
        """
        adcp_pref = self.config['Pref']['ADCP_PREF']
        buoy_pref = self.config['Pref']['BUOY_PREF']

        if self.target_col.startswith(adcp_pref):
            target_pref = adcp_pref
        elif self.target_col.startswith(buoy_pref):
            target_pref = buoy_pref
        else:
            raise IndexError
        # desc_str is of format:
        # <pred_target><forward hours>h_<time_res_min>m<input_data>_lb<look_back hours>h_<model_name><optional addition>
        desc_str = "{}{}h_{}m{}_lb{}h_{}{}".format(target_pref, self.pred_forward_hrs,
                                                   self.time_res_min, input_data_repr, self.look_back_hrs,
                                                   model_str_repr, addition)
        return desc_str

    def find_phys_col(self):
        """
        :return: the name of column holding the data of the physical model (WW3) for the relevant location (target col)
        """
        adcp_pref = self.config['Pref']['ADCP_PREF']
        buoy_pref = self.config['Pref']['BUOY_PREF']

        phy_deep_pref = self.config['Pref']['PHYS_DEEP_PREF']
        phys_shallow_pref = self.config['Pref']['PHYS_SHALLOW_PREF']

        if self.target_col.startswith(adcp_pref):
            phys_col = phy_deep_pref + "_hs"
        elif self.target_col.startswith(buoy_pref):
            phys_col = phys_shallow_pref + "_hs"
        else:
            raise IndexError
        return phys_col
