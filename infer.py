import struct
import random
import time
import numpy as np

def read_floats(f, n):
    return np.frombuffer(f.read(4 * n), dtype=np.float32).copy()

with open("weights.bin", "rb") as f:
    VOCAB    = struct.unpack("I", f.read(4))[0]
    chars    = [chr(struct.unpack("B", f.read(1))[0]) for _ in range(VOCAB)]
    N_LAYERS = struct.unpack("I", f.read(4))[0]
    EMBED    = struct.unpack("I", f.read(4))[0]
    HIDDEN   = struct.unpack("I", f.read(4))[0]
    CTX      = struct.unpack("I", f.read(4))[0]

    token_emb = read_floats(f, VOCAB * EMBED).reshape(VOCAB, EMBED)
    pos_emb   = read_floats(f, CTX * EMBED).reshape(CTX, EMBED)

    blocks = []
    for _ in range(N_LAYERS):
        rms1 = read_floats(f, EMBED)
        wq   = read_floats(f, EMBED * EMBED).reshape(EMBED, EMBED)
        wk   = read_floats(f, EMBED * EMBED).reshape(EMBED, EMBED)
        wv   = read_floats(f, EMBED * EMBED).reshape(EMBED, EMBED)
        wo   = read_floats(f, EMBED * EMBED).reshape(EMBED, EMBED)
        rms2 = read_floats(f, EMBED)
        w1   = read_floats(f, HIDDEN * EMBED).reshape(HIDDEN, EMBED)
        b1   = read_floats(f, HIDDEN)
        w2   = read_floats(f, EMBED * HIDDEN).reshape(EMBED, HIDDEN)
        b2   = read_floats(f, EMBED)
        blocks.append((rms1, wq, wk, wv, wo, rms2, w1, b1, w2, b2))

    rms_final = read_floats(f, EMBED)
    lm_head   = read_floats(f, VOCAB * EMBED).reshape(VOCAB, EMBED)

def rmsnorm(x, scale):
    inv = 1.0 / np.sqrt((x * x).mean() + 1e-5)
    return x * inv * scale

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def gelu(x):
    return x / (1.0 + np.exp(-1.702 * x))

SCALE = 1.0 / np.sqrt(EMBED)

def block_forward(seq, rms1, wq, wk, wv, wo, rms2, w1, b1, w2, b2, last_only=False):
    normed = np.array([rmsnorm(seq[t], rms1) for t in range(len(seq))])
    K = normed @ wk.T
    V = normed @ wv.T

    if last_only:
        q = normed[-1] @ wq.T
        scores = (K @ q) * SCALE
        w = softmax(scores)
        attn = w @ V
        x = seq[-1] + wo @ attn
        nx = rmsnorm(x, rms2)
        h = gelu(w1 @ nx + b1)
        return x + w2 @ h + b2
    else:
        T = len(seq)
        Q = normed @ wq.T
        scores = (Q @ K.T) * SCALE
        mask = np.triu(np.full((T, T), -1e9), 1)
        w = np.array([softmax(scores[t] + mask[t]) for t in range(T)])
        attn = w @ V
        x = seq + attn @ wo.T
        normed2 = np.array([rmsnorm(x[t], rms2) for t in range(T)])
        h = gelu(normed2 @ w1.T + b1)
        return x + h @ w2.T + b2

def forward(context):
    seq = np.array(context)
    for i in range(N_LAYERS - 1):
        seq = block_forward(seq, *blocks[i], last_only=False)
    last = block_forward(seq, *blocks[-1], last_only=True)
    normed = rmsnorm(last, rms_final)
    logits = lm_head @ normed
    return softmax(logits * 0.77)

def filtered_sample(probs, thresh, r):
    filtered = np.where(probs >= thresh, probs, 0.0)
    filtered /= filtered.sum()
    cdf = np.cumsum(filtered)
    return int(np.searchsorted(cdf, r))

def encode_prompt(prompt):
    char_to_idx = {c: i for i, c in enumerate(chars)}
    ctx = []
    for pos, ch in enumerate(prompt):
        idx = char_to_idx.get(ch, 0)
        ctx.append(token_emb[idx] + pos_emb[pos])
    return ctx

PROMPT   = "ROMEO:\nI "
N_TOKENS = 48
THRESH   = 0.05

randoms = [random.random() for _ in range(N_TOKENS)]

start = time.perf_counter()

context = encode_prompt(PROMPT)
ctx_len = len(PROMPT)
generated = []

for step in range(N_TOKENS):
    probs   = forward(context)
    r       = randoms[step]
    next_id = filtered_sample(probs, THRESH, r)
    generated.append(next_id)

    pos      = min(ctx_len, CTX - 1)
    next_vec = token_emb[next_id] + pos_emb[pos]
    context.append(next_vec)
    if len(context) > CTX:
        context = context[1:]
    ctx_len = min(ctx_len + 1, CTX)

elapsed = time.perf_counter() - start

output = PROMPT + "".join(chars[i] for i in generated) + "\n"
print(output)
print(f"NumPy inference: {elapsed:.2f}s  ({elapsed/N_TOKENS:.3f}s/token)")
