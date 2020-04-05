from torch import nn

class BoxDescriber(nn.Module):

    def __init__(self, feat_size, max_len, emb_size, rnn_num_layers, vocab_size):

        super(BoxDescriber, self).__init__()

        self.feat_size = feat_size
        self.max_len = max_len
        self.emb_size = emb_size
        self.rnn_num_layers = rnn_num_layers
        self.vocab_size = vocab_size
