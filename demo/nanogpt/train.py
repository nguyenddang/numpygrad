'''Train a char-nanogpt on the Shakespeare dataset using npg.'''

from model import GPT, GPTConfig

import pickle
import numpy as np

import npg
import time 
import os


#hyper parameters
batch_size = 12
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
max_iters = 2000
learning_rate = 1e-3
eval_iters = 10

model_args = dict(
    block_size=block_size,
    vocab_size=65,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout
)
cur_dir = os.path.join(os.path.dirname(__file__))
data_dir = os.path.join(cur_dir, 'data')
X_train = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
X_val = np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16)

def get_batch(split):
    if split=='train':
        data = X_train
    elif split=='val':
        data = X_val
    ix = np.random.randint(0, data.size - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i+1:i+block_size+1] for i in ix]).astype(np.int64)
    return npg.Tensor(x), npg.Tensor(y)

def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = npg.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = npg.mean(losses).item()
    return out
    

model = GPT(GPTConfig(**model_args))
optimiser = npg.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
t0 = time.time()
for i in range(max_iters):
    x, y = get_batch('train')
    _, loss = model(x, y)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    print(f"iter {i}, loss: {loss.item():4f} time: {dt*1000:.2f}ms")
    if i % 100 == 0:
        losses = estimate_loss()
        print(f"train loss: {losses['train']}, val loss: {losses['val']}")
