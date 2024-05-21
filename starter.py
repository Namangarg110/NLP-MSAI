import random

from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch
import json
import numpy as np


def load_corpus(file_name):
    answers = ['A', 'B', 'C', 'D']
    sequences = []

    with open(file_name) as json_file:
        json_list = list(json_file)
    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        base = result['fact1'] + ' [SEP] ' + result['question']['stem']
        ans = answers.index(result['answerKey'])

        obs = []
        for j in range(4):
            text = base + ' ' + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text, label])
        sequences.append(obs)
    return sequences


def format_data_for_BERT(sequences, tokenizer, hparams):
    tokenized_sequences = []
    tokenized_labels = []
    seq_lengths = []

    vocab = tokenizer.vocab
    max_len = 0
    for seq in sequences:
        for s, l in seq:
            tokenized = tokenizer.tokenize(s)
            ids = [vocab[t] for t in tokenized]
            target_seq = [vocab[str(l)], *ids]
            tokenized_sequences.append(target_seq)
            tokenized_labels.append(l)
            seq_len = len(target_seq)
            max_len = max(max_len, seq_len)
            seq_lengths.append(seq_len)

    out_sequences = np.zeros((len(tokenized_labels), max_len))
    out_masks = np.zeros((len(tokenized_labels), max_len))

    for i, s in enumerate(tokenized_sequences):
        for j in range(len(s)):
            out_sequences[i, j] = s[j]
            out_masks[i, j] = 1

    out_labels = np.zeros((len(tokenized_labels), 2))
    out_labels[:, 1] = tokenized_labels
    out_labels[:, 0] = 1 - out_labels[:, 1]

    out_sequences = torch.tensor(out_sequences, dtype=torch.int)
    out_labels = torch.tensor(out_labels, dtype=torch.float)
    out_masks = torch.tensor(out_masks, dtype=torch.int)

    hparams.max_len = max_len
    return out_sequences, out_labels, out_masks


class QBERT(torch.nn.Module):
    def __init__(self, pre_trained_model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
        self.BERT = BertModel.from_pretrained(pre_trained_model_name)
        self.linear = torch.rand((768, 2), requires_grad=True)

    def forward(self, x, mask):
        # grab the CLS embedding from the last hidden state
        x = self.BERT(x, attention_mask=mask).last_hidden_state[:, 0, :]
        x = x @ self.linear
        x = F.softmax(x, -1)
        return x


def evaluate_BERT(model, data, labels, masks, hparams, is_training=False):
    batch_size = hparams.batch_size
    batch_count = len(data) // batch_size
    model_loss = []
    model_acc = []
    if is_training:
        model.train()
    else:
        model.eval()

    for b in range(batch_count):
        if b % 100 == 0:
            print(f'\nBatch [{b}/{batch_count}]: ', end='')
        batch_start = b * batch_size
        batch_end = batch_start + batch_size
        batch_data = data[batch_start:batch_end]
        batch_labels = labels[batch_start:batch_end]
        batch_masks = masks[batch_start:batch_end]

        batch_output = model(batch_data, batch_masks)
        loss = F.binary_cross_entropy_with_logits(batch_output, batch_labels)
        preds = torch.argmax(batch_output, dim=-1)
        delta = [preds[i] == batch_labels[i, 1] for i in range(len(preds))]
        acc = sum(delta) / len(delta)
        if is_training:
            loss.backward()
            hparams.optimizer.step()
            hparams.optimizer.zero_grad()
        model_loss.append(loss.item())
        model_acc.append(acc.item())
        print('.', end='')
    print('done')

    return np.mean(model_loss), np.mean(model_acc)


def fine_tune_BERT(hparams):
    model = hparams.model
    epochs = hparams.epochs
    train_x, train_y, train_masks = format_data_for_BERT(load_corpus(hparams.train_corpus), model.tokenizer, hparams)
    valid_x, valid_y, valid_masks = format_data_for_BERT(load_corpus(hparams.valid_corpus), model.tokenizer, hparams)

    for e in range(epochs):
        print(f'\nStarting epoch {e + 1}:')
        train_loss, train_acc = evaluate_BERT(model, train_x, train_y, train_masks, hparams, is_training=True)
        valid_loss, valid_acc = evaluate_BERT(model, valid_x, valid_y, valid_masks, hparams)
        print(
            f'Epoch {e + 1} Results:\n\tTraining - Loss: {train_loss}, Accuracy: {train_acc}\n\tValidation - Loss: {valid_loss}, Accuracy: {valid_acc}')


def test_fine_tuned_BERT(hparams):
    model = hparams.model
    test_x, test_y, test_masks = format_data_for_BERT(load_corpus(hparams.test_corpus), model.tokenizer, hparams)
    test_loss, test_acc = evaluate_BERT(model, test_x, test_y, test_masks, hparams)
    print(f'Test Results: Loss: {test_loss}, Accuracy: {test_acc}')


def BERT_QA(hparams):
    torch.set_default_device(hparams.device)
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    hparams.model = QBERT("bert-base-uncased")
    hparams.optimizer = optim.Adam(hparams.model.parameters(), lr=hparams.lr)
    hparams.train_corpus = 'train_complete.jsonl'
    hparams.valid_corpus = 'dev_complete.jsonl'
    hparams.test_corpus = 'test_complete.jsonl'
    fine_tune_BERT(hparams)
    test_fine_tuned_BERT(hparams)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-model_type', type=str, default='BERT')
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-lr', type=float, default=3e-5)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-seed', type=int, default=1337)
    parser.add_argument('-max_len', type=int, default=40)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_type == 'BERT':
        BERT_QA(args)
    # add GPT case here


if __name__ == "__main__":
    main()
