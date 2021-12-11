# BEGIN - DO NOT CHANGE THESE IMPORTS OR IMPORT ADDITIONAL PACKAGES.
from keras.initializers import Constant
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
import torch
import torch.nn as nn
# END - DO NOT CHANGE THESE IMPORTS OR IMPORT ADDITIONAL PACKAGES.


class ClassificationModel(nn.Module):
    def __init__(self, action_vocab_size, situation_vocab_size, intent_vocab_size, action_embedding_matrix, situation_embedding_matrix, intent_embedding_matrix,
                 embedding_dim, hidden_dim,
                 num_layers=3, bidirectional=True):
        super().__init__()

        self.embedding_action = nn.Embedding(
            action_embedding_matrix.shape[0], action_embedding_matrix.shape[1])
        self.embedding_action.weight = nn.Parameter(
            torch.from_numpy(action_embedding_matrix).float())
        self.embedding_action.weight.requires_grad = False

        self.embedding_situation = nn.Embedding(
            situation_embedding_matrix.shape[0], situation_embedding_matrix.shape[1])
        self.embedding_situation.weight = nn.Parameter(
            torch.from_numpy(situation_embedding_matrix).float())
        self.embedding_situation.weight.requires_grad = False

        self.embedding_intent = nn.Embedding(
            intent_embedding_matrix.shape[0], intent_embedding_matrix.shape[1])
        self.embedding_intent.weight = nn.Parameter(
            torch.from_numpy(intent_embedding_matrix).float())
        self.embedding_intent.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            bidirectional=bidirectional, batch_first=True)

        self.lstm2 = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                             bidirectional=bidirectional, batch_first=True)

        self.lstm3 = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                             bidirectional=bidirectional, batch_first=True)
        hidden_dim_size = hidden_dim * (2 if bidirectional else 1)
        self.linear = nn.Linear(hidden_dim_size*3, hidden_dim_size*2)
        self.linear2 = nn.Linear(hidden_dim_size*2, 1)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def forward(self, x1, x2, x3):
        embeds1 = self.embedding_action(x1)
        embeds2 = self.embedding_situation(x2)
        embeds3 = self.embedding_intent(x3)

        def lstm_forward(lstm, embed):
            lstm_out, (hidden, cell) = lstm(embed)
            if(self.bidirectional):
                state1, state2 = torch.chunk(lstm_out, 2, dim=2)
                lstm_out = torch.cat(
                    (state1[:, -1, :], state2[:, 0, :]), dim=1)
            else:
                lstm_out = lstm_out[:, -1, :]
            return lstm_out

        out1 = lstm_forward(self.lstm, embeds1)
        out2 = lstm_forward(self.lstm2, embeds2)
        out3 = lstm_forward(self.lstm3, embeds3)
        lstm_out = torch.concat([out1, out2, out3], 1)

        lstm_out = self.linear(lstm_out)
        lstm_out = self.linear2(lstm_out)
        lstm_out = self.sigmoid(lstm_out)

        return lstm_out
