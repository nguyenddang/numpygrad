'''Train a char-nanogpt on the Shakespeare dataset using npg.'''

from model import GPT, GPTConfig

import pickle
import numpy as np

import npg
import time 


#hyper parameters
batch_size = 12
block_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
max_iters = 2000
learning_rate = 1e-3

model_args = dict(
    block_size=block_size,
    vocab_size=65,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout
)

X_train = np.fromfile('data/train.bin', dtype=np.uint16)
X_val = np.fromfile('data/val.bin', dtype=np.uint16)

def get_batch(split):
    if split=='train':
        data = X_train
    elif split=='val':
        data = X_val
    ix = np.random.randint(0, data.size - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i+1:i+block_size+1] for i in ix]).astype(np.int64)
    return npg.Tensor(x), npg.Tensor(y)
    

model = GPT(GPTConfig(**model_args))
optimiser = npg.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
t0 = time.time()
for i in range(max_iters):
    x, y = get_batch('train')
    logits, loss = model(x, y)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    print(f"iter {i}, loss: {loss.item():4f} time: {dt*1000:.2f}ms")
