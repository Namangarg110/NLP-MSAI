import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torcheval.metrics import Perplexity
from transformers import GPT2TokenizerFast


def ensure_directory_path_exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)


def read_corpus(filename, tokenizer):
    seq = []
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return seq


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x.int())


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=4096, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                    self.eta_min + ((lr - self.eta_min) / 2) *
                    (
                            np.cos(
                                np.pi *
                                (self._cycle_counter % self._updated_cycle_len) /
                                self._updated_cycle_len
                            ) + 1
                    )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=max_sequence_length, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, trg_vocab, max_sequence_length, d_model, N, heads, dropout):
        super().__init__()
        self.decoder = Decoder(trg_vocab, max_sequence_length, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, trg, trg_mask):
        # print("DECODER")
        d_output = self.decoder(trg, trg_mask)
        output = self.out(d_output)
        return output


def get_model(opt, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(trg_vocab, opt.seqlen, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)

    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    return model


def evaluate(model, data, opt, is_training):
    if is_training:
        model.train()
    else:
        model.eval()
    seq_len = opt.seqlen
    batch_size = opt.batch_size
    batch_step = seq_len * batch_size
    optimizer = opt.optimizer
    nopeak_mask = opt.mask
    corpora = torch.tensor([*data], device='cuda')
    data_len = len(corpora)
    batch_count = data_len // batch_step
    ppl_eval = opt.ppl_eval
    batch_indices = np.arange(batch_count)
    np.random.shuffle(batch_indices)
    batch_losses = []
    batch_ppls = []
    for i in range(batch_count):
        if i % opt.printevery == 0:
            print(f'\nBatch {i}/{batch_count}: ', end='')
        bi = batch_indices[i]
        batch_start = bi * batch_step
        batch_end = batch_start + batch_step
        batch = corpora[batch_start:batch_end].view((batch_size, seq_len))
        label = corpora[batch_start + 1:min(batch_end + 1, data_len)].view((batch_size, seq_len))
        logits = model.forward(batch, nopeak_mask)
        loss = F.cross_entropy(logits.movedim(-2, -1), label)
        if is_training:
            loss.backward()
            optimizer.step()
        ppl_eval.update(logits, label)
        ppl = ppl_eval.compute()
        batch_losses.append(loss.item())
        batch_ppls.append(ppl.item())
        print('.', end='')
        if (i + 1) % opt.printevery == 0:
            print(f'<L:{loss} P:{ppl}>', end='')
    print('\ndone')

    return batch_losses, batch_ppls


def train_model(model, opt):
    print("training model...")
    train_corpora = torch.tensor([*opt.train]).cuda()
    valid_corpora = torch.tensor([*opt.valid]).cuda()
    for e in range(opt.epochs):
        print(f'Epoch {e}:')
        epoch_loss, epoch_ppl = evaluate(model, train_corpora, opt, True)
        valid_loss, valid_ppl = evaluate(model, valid_corpora, opt, False)
        print(f'\nEpoch {e} Result:')
        print(f'\n\tTraining\n\t\tMean Loss: {np.mean(epoch_loss)}\n\t\tMean Perplexity: {np.mean(epoch_ppl)}')
        print(f'\n\tValidation\n\t\tMean Loss: {np.mean(valid_loss)}\n\t\tMean Perplexity: {np.mean(valid_ppl)}')

        if opt.savename is not None:
            torch.save(model, f'{opt.savename}/model.pt')

        test_model(model, opt, e)


def test_model(model, opt, epoch):
    print("testing model...")
    test_corpora = torch.tensor([*opt.test]).cuda()
    test_loss, test_ppl = evaluate(model, test_corpora, opt, False)
    print(f'Epoch {epoch}:\n\tTest\n\t\tMean Loss: {np.mean(test_loss)}\n\t\tMean Perplexity: {np.mean(test_ppl)}')


def main():
    random.seed(10)

    opt = argparse.Namespace()
    opt.no_cuda = False
    opt.SGDR = False
    opt.epochs = 20
    opt.d_model = 512
    opt.n_layers = 6
    opt.heads = 8
    opt.dropout = 0.1
    opt.batchsize = 16
    opt.printevery = 100
    opt.lr = 0.00001
    opt.seqlen = 512
    opt.threshold = 3
    opt.savename = None
    opt.loadname = None
    opt.tied = 1
    opt.dir_name = 'model'
    opt.norm = 2.0
    opt.train = None
    opt.valid = None
    opt.test = None
    opt.train_len = None
    opt.log_file = None
    opt.time_name = None
    opt.norm = 2.0

    opt.verbose = False

    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.device = torch.device("cuda:0")

    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % opt.dir_name
    ensure_directory_path_exists(dir_name)
    source_name = sys.argv[0].split('\\')[-1]
    dir_name = dir_name + "//"
    opt.dir_name = dir_name
    # shutil.copy(source_name, dir_name + source_name)
    opt.log_file = dir_name + "log_file.txt"

    print(str(opt))

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    data_dir = '.'

    opt.train = read_corpus(f'{data_dir}/wiki2.train.txt', tokenizer)
    opt.valid = read_corpus(f'{data_dir}/wiki2.valid.txt', tokenizer)
    opt.test = read_corpus(f'{data_dir}/wiki2.test.txt', tokenizer)

    opt.savename = './saved/model'
    opt.batch_size = opt.batchsize
    opt.vocab_size = 50257
    opt.trg_pad = 0
    opt.mask = torch.broadcast_to(torch.triu(torch.ones((opt.seqlen, opt.seqlen))).T,
                                  (opt.batch_size, opt.seqlen, opt.seqlen)).cuda()
    opt.ppl_eval = Perplexity(device='cuda', ignore_index=opt.trg_pad)

    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()

    model = get_model(opt, opt.vocab_size)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    text = 'total params: %d' % params
    print(text)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1

    train_model(model, opt)
    test_model(model, opt, -1)


if __name__ == "__main__":
    main()
