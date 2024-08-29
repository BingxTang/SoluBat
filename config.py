import configparser


class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # General Parameters
        self.Seed = conf.getint("General", "seed")
        self.Num_epochs = conf.getint("General", "num_epochs")
        self.Logdir = conf.get("General", "logdir")
        self.Batch_size = conf.getint("General", "batch_size")

        # Database Parameters
        self.DataFolder = conf.get("Database", "dataFolder")
        self.ResultFolder = conf.get("Database", "resultfolder")
        self.ModelFolder = conf.get("Database", "modelfolder")
        self.Train_Database = conf.get("Database", "train_database")
        self.Test_Database = conf.get("Database", "test_database")
        self.Checkpoints = conf.get("Database", "checkpoints")

        # Model Parameters
        self.learning_rate = conf.getfloat("Model", "learning_rate")
        self.min_len = conf.getint("Model", "min_len")
        self.max_len = conf.getint("Model", "max_len")

        self.mam_d_model = conf.getint("Model", "mam_d_model")
        self.mam_n_layer = conf.getint("Model", "mam_n_layer")
        self.mam_d_intermediate = conf.getint("Model", "mam_d_intermediate")
        self.mam_vocab_size = conf.getint("Model", "mam_vocab_size")
        self.mam_rms_norm = conf.getboolean("Model", "mam_rms_norm")
        self.mam_residual_in_fp32 = conf.getboolean("Model", "mam_residual_in_fp32")
        self.mam_fused_add_norm = conf.getboolean("Model", "mam_fused_add_norm")
        self.mam_dropout_prob = conf.getfloat("Model", "mam_dropout_prob")

        self.rnn_in_channels = conf.getint("Model", "rnn_in_channels")
        self.rnn_n_layers = conf.getint("Model", "rnn_n_layers")
        self.rnn_conv1d_feature_size = conf.getint("Model", "rnn_conv1d_feature_size")
        self.rnn_conv1d_kernel_size = conf.getint("Model", "rnn_conv1d_kernel_size")
        self.rnn_avgpool1d_kernel_size = conf.getint("Model", "rnn_avgpool1d_kernel_size")
        self.rnn_gru_hidden_size = conf.getint("Model", "rnn_gru_hidden_size")
        self.rnn_fully_connected_layer_size = conf.getint("Model", "rnn_fully_connected_layer_size")
        self.rnn_dropout_prob = conf.getfloat("Model", "rnn_dropout_prob")