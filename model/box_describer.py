import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BoxDescriber(nn.Module):

    def __init__(self, feat_size, hidden_size, max_len, emb_size, rnn_num_layers, vocab_size,
                 fusion_type='init_inject', pad_idx=0, start_idx=1, end_idx=2):

        assert fusion_type in {'init_inject', 'merge'}, "only init_inject and merge is supported"

        super(BoxDescriber, self).__init__()

        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.emb_size = emb_size
        self.rnn_num_layers = rnn_num_layers
        self.vocab_size = vocab_size
        self.fusion_type = fusion_type
        self.special_idx = {
            '<pad>':pad_idx,
            '<bos>':start_idx,
            '<eos>':end_idx
        }

        self.embedding_layer = nn.Embedding(vocab_size, emb_size)

        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=rnn_num_layers, batch_first=True)

        self.feature_project_layer = nn.Sequential(
            nn.Linear(feat_size, emb_size),
            nn.ReLU()
        )

        if fusion_type == 'init_inject':
            self.fc_layer = nn.Linear(hidden_size, vocab_size)
        else: # merge
            self.fc_layer = nn.Linear(hidden_size + emb_size, vocab_size)

    def init_hidden(self, batch_size, device):

        h0 = torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.rnn_num_layers, batch_size, self.hidden_size).to(device)

        return h0, c0

    def forward_train(self, feat, cap_gt, cap_lens):
        """

        :param feat: (batch_size, feat_size)
        :param cap_gt: (batch_size, max_len)
        :param cap_lens: (batch_size,)
        :return: predicts (batch_size, max_len, vocab_size)
        """

        batch_size = feat.shape[0]
        device = feat.device

        word_emb = self.embedding_layer(cap_gt)  # (batch_size, max_len, embed_size)
        feat_emb = self.feature_project_layer(feat)  # (batch_size, embed_size)

        h0, c0 = self.init_hidden(batch_size, device)
        if self.fusion_type == 'inject_init':
            h0, c0 = self.rnn(feat_emb.unsqueeze(1), (h0, c0))  # first input projected feat emb to rnn

        rnn_input_pps = pack_padded_sequence(word_emb, lengths=cap_lens, batch_first=True, enforce_sorted=False)

        rnn_output_pps, _ = self.rnn(rnn_input_pps, (h0, c0))

        rnn_output, _ = pad_packed_sequence(rnn_output_pps, batch_first=True, total_length=self.max_len)

        if self.fusion_type == 'merge':
            feat_emb = feat_emb[:, None, :].expand(batch_size, self.max_len, self.emb_size)
            rnn_output = torch.cat([rnn_output, feat_emb], dim=-1)  # (batch_size, max_len, hidden_size + emb_size)

        predicts = self.fc_layer(rnn_output)

        return predicts

    def forward_test(self, feat):
        """Greedy inference for the sake of speed

        :param feat: (batch_size, feat_size)
        :return: predicts (batch_size, max_len)
        """
        batch_size = feat.shape[0]
        device = feat.device

        feat_emb = self.feature_project_layer(feat)  # (batch_size, embed_size)

        predicts = torch.ones(batch_size, self.max_len+1, dtype=torch.long).to(device) * self.special_idx['<pad>']
        predicts[:, 0] = torch.ones(batch_size, dtype=torch.long).to(device) * self.special_idx['<bos>']
        keep = torch.arange(batch_size,)  # keep track of unfinished sequences

        h, c = self.init_hidden(batch_size, device)
        if self.fusion_type == 'inject_init':
            h, c = self.rnn(feat_emb.unsqueeze(1), (h, c))  # first input projected feat emb to rnn

        for i in range(self.max_len):
            word_emb = self.embedding_layer(predicts[keep, i])  # (valid_batch_size, embed_size)

            _, (h, c) = self.rnn(word_emb.unsqueeze(1), (h, c))  # (num_layers, valid_batch_size, hidden_size)

            if self.fusion_type == 'inject_init':
                rnn_output = h[-1]
            else: # merge
                rnn_output = torch.cat([h[-1], feat_emb], dim=-1)  # (valid_batch_size, hidden_size + emb_size)

            pred = self.fc_layer(rnn_output)  # (valid_batch_size, vocab_size)

            predicts[keep, i+1] = pred.log_softmax(dim=-1).argmax(dim=-1)

            keep = keep[predicts[keep, i+1] != self.special_idx['<eos>']]  # update unfinished indices
            if keep.nelement() == 0:  # stop if all finished
                break
            else:
                h = h[:, predicts[keep, i+1] != self.special_idx['<eos>'], :]
                c = c[:, predicts[keep, i+1] != self.special_idx['<eos>'], :]

        return predicts

    def forward(self, feat, cap_gt=None, cap_lens=None):

        if isinstance(cap_gt, list) and isinstance(cap_lens, list):
            cap_gt = torch.cat(cap_gt, dim=0)
            cap_lens = torch.cat(cap_lens, dim=0)
            assert feat.shape[0] == cap_gt.shape[0] and feat.shape[0] == cap_lens.shape[0]

        if self.training:
            assert cap_gt is not None and cap_lens is not None, "cap_gt and cap_lens should not be None during training"
            cap_gt = cap_gt[:, :-1]  # '<eos>' does not include in input
            cap_lens = torch.clamp(cap_lens - 1, min=0)
            return self.forward_train(feat, cap_gt, cap_lens)
        else:
            return self.forward_test(feat)
