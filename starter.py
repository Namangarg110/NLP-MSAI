from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim

import torch
import math
import time
import sys
import json
import numpy as np


def load_corpus(file_name, tokenizer):
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
            text = base + result['question']['choices'][j]['text'] + ' [SEP]'
            if j == ans:
                label = 1
            else:
                label = 0
            obs.append([text, label])
        sequences.append(obs)

    tokenized_sequences = []

    for seq in sequences:
        tokenized_sequences.append([(tokenizer.tokenize(s[0]), s[1]) for s in seq])

    return tokenized_sequences


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    train = load_corpus('train_complete.jsonl', tokenizer)
    test = load_corpus('dev_complete.jsonl', tokenizer)
    valid = load_corpus('test_complete.jsonl', tokenizer)

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    linear = torch.rand(768, 2)


if __name__ == "__main__":
    main()
