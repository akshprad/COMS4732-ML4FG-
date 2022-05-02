from typing import List


class Welcome1:
    batch_norm_decay: float
    batch_normalization: bool
    batch_sizes: List[int]
    buckets: List[int]
    comm_nn_lstm: bool
    comm_nn_lstm_alternate: bool
    comm_nn_lstm_layers: int
    comm_nn_lstm_size: int
    comm_nn_lstm_steps: int
    comm_nn_conv: bool
    comm_nn_fc: bool
    comm_nn_fc_final: bool
    conv_params: List[List[int]]
    data_dir: str
    dataset_type: str
    dropout_keep_prob: int
    early_stopping_delay: int
    event_type: str
    exonic_seq_length: int
    features: List[str]
    graphing_frequency: int
    hidden_units: List[int]
    init_scale_lstm: float
    init_scale_conv: float
    init_scale_fc: float
    init_scale_fc_final: float
    intronic_seq_length: int
    l2_decay: float
    learning_rate: float
    lr_decay: bool
    lr_decay_rate: float
    lr_decay_step: List[int]
    make_rundir: bool
    max_grad_norm: int
    max_sgd_steps: int
    n_validation_files: int
    n_validation_steps: int
    num_epochs: int
    num_outputs: int
    out_lstm_size: int
    output_lstm: bool
    output_lstm_bidirectional: bool
    sort_ss_by_position: bool
    train_dir: str
    validation_frequency: int

    def __init__(self, batch_norm_decay: float, batch_normalization: bool, batch_sizes: List[int], buckets: List[int], comm_nn_lstm: bool, comm_nn_lstm_alternate: bool, comm_nn_lstm_layers: int, comm_nn_lstm_size: int, comm_nn_lstm_steps: int, comm_nn_conv: bool, comm_nn_fc: bool, comm_nn_fc_final: bool, conv_params: List[List[int]], data_dir: str, dataset_type: str, dropout_keep_prob: int, early_stopping_delay: int, event_type: str, exonic_seq_length: int, features: List[str], graphing_frequency: int, hidden_units: List[int], init_scale_lstm: float, init_scale_conv: float, init_scale_fc: float, init_scale_fc_final: float, intronic_seq_length: int, l2_decay: float, learning_rate: float, lr_decay: bool, lr_decay_rate: float, lr_decay_step: List[int], make_rundir: bool, max_grad_norm: int, max_sgd_steps: int, n_validation_files: int, n_validation_steps: int, num_epochs: int, num_outputs: int, out_lstm_size: int, output_lstm: bool, output_lstm_bidirectional: bool, sort_ss_by_position: bool, train_dir: str, validation_frequency: int) -> None:
        self.batch_norm_decay = batch_norm_decay
        self.batch_normalization = batch_normalization
        self.batch_sizes = batch_sizes
        self.buckets = buckets
        self.comm_nn_lstm = comm_nn_lstm
        self.comm_nn_lstm_alternate = comm_nn_lstm_alternate
        self.comm_nn_lstm_layers = comm_nn_lstm_layers
        self.comm_nn_lstm_size = comm_nn_lstm_size
        self.comm_nn_lstm_steps = comm_nn_lstm_steps
        self.comm_nn_conv = comm_nn_conv
        self.comm_nn_fc = comm_nn_fc
        self.comm_nn_fc_final = comm_nn_fc_final
        self.conv_params = conv_params
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.dropo  ut_keep_prob = dropout_keep_prob
        self.early_stopping_delay = early_stopping_delay
        self.event_type = event_type
        self.exonic_seq_length = exonic_seq_length
        self.features = features
        self.graphing_frequency = graphing_frequency
        self.hidden_units = hidden_units
        self.init_scale_lstm = init_scale_lstm
        self.init_scale_conv = init_scale_conv
        self.init_scale_fc = init_scale_fc
        self.init_scale_fc_final = init_scale_fc_final
        self.intronic_seq_length = intronic_seq_length
        self.l2_decay = l2_decay
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.make_rundir = make_rundir
        self.max_grad_norm = max_grad_norm
        self.max_sgd_steps = max_sgd_steps
        self.n_validation_files = n_validation_files
        self.n_validation_steps = n_validation_steps
        self.num_epochs = num_epochs
        self.num_outputs = num_outputs
        self.out_lstm_size = out_lstm_size
        self.output_lstm = output_lstm
        self.output_lstm_bidirectional = output_lstm_bidirectional
        self.sort_ss_by_position = sort_ss_by_position
        self.train_dir = train_dir
        self.validation_frequency = validation_frequency
