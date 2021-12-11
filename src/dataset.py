import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
PADDING_VALUE = 0
UNK_VALUE = 1


def split_train_val_test(df, props=[.8, .1, .1]):
    assert round(sum(props), 2) == 1 and len(props) >= 2
    train_df, test_df, val_df = None, None, None
    # split df into train, test, val
    # df.sample(frac=1).reset_index(drop=True)
    train_len = int(props[0]*df.shape[0])
    test_len = int(props[1]*df.shape[0])
    print(train_len, test_len)
    train_df = df.iloc[:train_len].reset_index(drop=True)
    test_df = df.iloc[train_len:train_len+test_len].reset_index(drop=True)
    val_df = df.iloc[train_len+test_len:].reset_index(drop=True)
    return train_df, val_df, test_df


def generate_vocab_map(df, cutoff=2):
    vocab = {"": PADDING_VALUE, "UNK": UNK_VALUE}
    reversed_vocab = {PADDING_VALUE: "", UNK_VALUE: "UNK"}

    ## YOUR CODE STARTS HERE (~5-15 lines of code) ##
    # hint: start by iterating over df["tokenized"]
    # count number of items in vocab and add them
    # to vocab if they appear more than cutoff times

    token_counts = Counter()
    for tokens in df["action_tokenized"]:
        token_counts.update(tokens)
    token_counts = {x: count for x,
                    count in token_counts.items() if count >= cutoff}
    for token in token_counts:
        vocab.setdefault(token, len(vocab))
    for token in vocab:
        reversed_vocab.setdefault(vocab[token], token)
    return vocab, reversed_vocab


class MoralStoriesDataset(Dataset):

    def __init__(self, vocab, df, max_length=50):
        self.df = df
        self.max_length = max_length
        self.vocab = vocab
        return

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        curr_label = None
        action_sentence = self.df.iloc[index]["action_tokenized"]
        situation_sentence = self.df.iloc[index]["situation_tokenized"]
        norm_sentence = self.df.iloc[index]["norm_tokenized"]
        action_sentence_tensor = torch.LongTensor(
            [self.vocab.get(word, UNK_VALUE) for word in action_sentence])
        situation_sentence_tensor = torch.LongTensor(
            [self.vocab.get(word, UNK_VALUE) for word in situation_sentence])
        norm_sentence_tensor = torch.LongTensor(
            [self.vocab.get(word, UNK_VALUE) for word in norm_sentence])

        curr_label = self.df.iloc[index]["label"]
        ## YOUR CODE ENDS HERE ##
        return action_sentence_tensor, situation_sentence_tensor, norm_sentence_tensor, curr_label


def collate_fn(batch, padding_value=PADDING_VALUE):
    padded_tokens, y_labels = None, None
    action_sequences = [item[0] for item in batch]
    situation_sequences = [item[1] for item in batch]
    norm_sequences = [item[2] for item in batch]

    y_labels = torch.FloatTensor([item[3] for item in batch])

    action_padded_tokens = pad_sequence(
        action_sequences, padding_value=padding_value, batch_first=True)
    situations_padded_tokens = pad_sequence(
        situation_sequences, padding_value=padding_value, batch_first=True)
    norm_padded_tokens = pad_sequence(
        norm_sequences, padding_value=padding_value, batch_first=True)
    return action_padded_tokens, situations_padded_tokens, norm_padded_tokens, y_labels
