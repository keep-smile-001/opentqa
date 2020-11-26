from opentqa.core.base_cfgs import Configurations

class Cfgs(Configurations):
    def __init__(self):
        super(Cfgs, self).__init__()
        self.max_ans = 4
        self.classifer_dropout_r = 0.1  # owing to the different distributions between val and test, val:0.1 better; test:0.2 better;
        self.dropout_r = 0.2
        self.dia_feat_size = 2048       # the feature of the diagram
        self.flat_out_size = 2048       # flatten hidden size
        self.flat_mlp_size = 512        # flatten one tensor to 512-dimensional tensor
        self.flat_glimpse = 1
        self.glimpse = 1
        self.hidden_size = 1024
        self.lr_base = 0.0001
        self.lr_decay_r = 0.25           # decay rate test:0.1 better;
        self.max_opt_token = 5          # the maximum token of a option
        self.max_que_token = 15         # the maximum token of a [question] test: 10; better;
        self.max_sent = 15               # the maximum number of sentences within a paragraph test: 5 better;
        self.max_sent_token = 15        # the maximum number of tokens per sentence; test:20 better;
        self.k_times = 3
        self.ba_hidden_size = self.k_times * self.hidden_size
        self.span_width = 2
        self.p_times = 2
        self.max_seq_length = 20
        self.batch_size = 2
        self.optim = 'bert'
        self.from_scratch = False
