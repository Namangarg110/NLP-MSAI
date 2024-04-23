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
from transformers import GPT2TokenizerFast

def read_corpus(filename,tokenizer):
    seq = []
    with open(filename,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
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
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, float('-inf'))
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
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
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
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
                        ((self._cycle_counter) % self._updated_cycle_len) /
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
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        # x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        # src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        # self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        # e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, src, src_mask, trg_mask)
        output = self.out(d_output)
        return output

def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
       
    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model
def create_batches(data, batch_size, seq_len, pad_token=0):
    # Step 1: Cut the data into chunks of `seq_len`
    num_batches = len(data) // (batch_size * seq_len)
    # Trim the data to fit exactly into batches
    data = data[:num_batches * batch_size * seq_len]    
    # Step 2: Reshape data to batch_size rows
    data = torch.tensor(data).view(batch_size, -1)
    # Step 3: Create batches
    batches = []
    for i in range(0, data.size(1), seq_len):
        if i + seq_len <= data.size(1):
            batch = data[:, i:i+seq_len]
            batch_target = data[:, i+1:i+seq_len+1]
            batches.append((batch, batch_target))
    
    return batches
def create_mask(seq, pad_token):
    # Create a mask to hide padding
    pad_mask = (seq != pad_token).unsqueeze(-2)

    # Create a mask to hide future tokens (upper triangular)
    nopeak_mask = torch.triu(torch.ones((1, seq.size(-1), seq.size(-1)), device=seq.device), diagonal=1).bool()
    
    return pad_mask & ~nopeak_mask  # Combine the pad and no peak masks
    
def train_model(model, opt):
    
    print("training model...")
    model.train()
    train_data = create_batches(opt.train, opt.batchsize, opt.seqlen, opt.src_pad)
    valid_data = create_batches(opt.valid, opt.batchsize, opt.seqlen, opt.src_pad)
    print("Created Batches")

    for epoch in range(opt.epochs):
        total_loss = 0
        total_tokens = 0 

        print(f"Epoch {epoch+1}")
        for i, (input_seq, target_seq) in enumerate(train_data):
            input_seq = input_seq.to(opt.device)  # B, T, T 
            target_seq = target_seq.to(opt.device)#  B, T, T but one index shifted to the right
            
            # Create masks
            src_mask = create_mask(input_seq, opt.src_pad) # B, T, T
            trg_mask = create_mask(target_seq, opt.trg_pad) # B, T, T 
            
            # Reset gradients
            opt.optimizer.zero_grad()
            
            # Forward pass
            output = model(input_seq, target_seq, src_mask, trg_mask) # B, T, vocab_size
            
            # Calculate loss
            loss = F.cross_entropy(output.view(-1, opt.vocab_size), target_seq.view(-1), ignore_index=opt.trg_pad) # output.view -> B*T-1, vocab_size, target_seq.view(-1) -> B*T-1
            total_loss += loss.item() * target_seq.numel()  # Multiply the loss by the number of non-pad tokens
            total_tokens += target_seq.numel()
            
            # Backward pass and update
            loss.backward()
            opt.optimizer.step()
            
            if (i + 1) % opt.printevery == 0:
                print(f'Batch {i+1}, Loss: {loss.item()}')
        
        # Calculate average loss per epoch
        avg_loss = total_loss / total_tokens
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss}, Perplexity: {calculate_perplexity(avg_loss)}')
        
        # Test the model
        test_model(model, opt, valid_data)
        model.train()
    
    torch.save(model.state_dict(), opt.savename)

    # write code to:
    #  1. create a nopeak mask
    #  2. feed training data to the model in batches
    #  3. send the indices of training tokens to the GPU
    #  4. linearize the predictions and compute the loss against ground truth
    #     (you can use F.cross_entropy or write your own code)
    #  5. calculate and apply the gradients with loss.backward() and optimizer.step()
    #  6. report intermediate trainining perplexity
    #  7. generate a test perplexity once per training epoch by calling test_model()
    #  8. save model weights to file specified in opt.savename
    #  SEE trainer.py for examples of each of the above
    

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss))

def test_model(model, opt, data_loader):
    print("testing model...")
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for input_seq, target_seq in data_loader:
            input_seq = input_seq.to(opt.device)
            target_seq = target_seq.to(opt.device)
            src_mask = create_mask(input_seq, opt.src_pad)
            trg_mask = create_mask(target_seq, opt.trg_pad)

            output = model(input_seq, target_seq, src_mask, trg_mask) 
            output_flat = output.view(-1, opt.vocab_size)
            target_flat = target_seq.view(-1)  

            loss = F.cross_entropy(output_flat, target_flat, ignore_index=opt.trg_pad)
            total_loss += loss.item() * target_seq.numel()
            total_tokens += target_seq.numel()

    avg_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(avg_loss)
    print(f"Test Perplexity: {perplexity}")
    model.train()
    return perplexity

def main():
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=16)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str)    
    parser.add_argument('-loadname', type=str)    
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str,default='model')
    parser.add_argument('-norm', type=float, default=2.0)
                
    opt = parser.parse_args()
    opt.verbose = False    
    
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.device = torch.device("cuda:0")
    
    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    source_name = sys.argv[0]
    dir_name = dir_name + "//"
    opt.dir_name = dir_name
    os.makedirs(source_name+dir_name,exist_ok=True)
    shutil.copy(source_name, source_name+dir_name)
    opt.log_file = dir_name + "log_file.txt"
    
    print(str(opt))
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.train = read_corpus('wiki2.train.txt',tokenizer) # 2411664 tokens
    opt.valid = read_corpus('wiki2.valid.txt',tokenizer)
    opt.test = read_corpus('wiki2.test.txt',tokenizer)
    obs = len(opt.train)
    opt.vocab_size = 50257
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()
    
    model = get_model(opt,opt.vocab_size,opt.vocab_size)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    text = 'total params: %d' % (params)
    print(text)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1
    opt.src_pad = 0
    opt.trg_pad = 0
            
    train_model(model,opt)
    test_data = create_batches(opt.test, opt.batchsize, opt.seqlen, opt.src_pad)
    test_model(model,opt,test_data)
        
if __name__ == "__main__":
    main()        