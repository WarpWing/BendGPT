# bend only has 24-bit floats (f24) so I can't be as precise as I want :(
type FVec:
  Nil
  Cons { head: f24, ~tail: FVec }

type FMatrix:
  Nil
  Cons { row: FVec, ~tail: FMatrix }

type Seq:
  Nil
  Cons { vec: FVec, ~tail: Seq }

type Weights:
  Mk {
    wq: FMatrix, wk: FMatrix, wv: FMatrix, wo: FMatrix,
    w1: FMatrix, b1: FVec,
    w2: FMatrix, b2: FVec,
    whead: FMatrix
  }

type Tokens:
  Nil
  Cons { id: u24, ~tail: Tokens }

def fvec_dot(a: FVec, b: FVec) -> f24:
  fold a with b:
    case FVec/Nil:
      return 0.0
    case FVec/Cons:
      match b:
        case FVec/Nil:
          return 0.0
        case FVec/Cons:
          return a.head * b.head + a.tail(b.tail)

def fvec_add(a: FVec, b: FVec) -> FVec:
  match a:
    case FVec/Nil:
      return b
    case FVec/Cons:
      match b:
        case FVec/Nil:
          return a
        case FVec/Cons:
          return FVec/Cons { head: a.head + b.head, tail: fvec_add(a.tail, b.tail) }

def fvec_scale(v: FVec, s: f24) -> FVec:
  fold v with s:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      return FVec/Cons { head: v.head * s, tail: v.tail(s) }

def fvec_relu(v: FVec) -> FVec:
  fold v:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      if v.head > 0.0:
        return FVec/Cons { head: v.head, tail: v.tail }
      else:
        return FVec/Cons { head: 0.0, tail: v.tail }

def fvec_max(v: FVec, current: f24) -> f24:
  fold v with current:
    case FVec/Nil:
      return current
    case FVec/Cons:
      if v.head > current:
        return v.tail(v.head)
      else:
        return v.tail(current)

def fvec_sum(v: FVec) -> f24:
  fold v:
    case FVec/Nil:
      return 0.0
    case FVec/Cons:
      return v.head + v.tail

def fvec_shift(v: FVec, s: f24) -> FVec:
  fold v with s:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      return FVec/Cons { head: v.head - s, tail: v.tail(s) }

# bend has no exp() builtin so we use a taylor series
# 6 terms is enough given f24 only has ~3 decimal digits of precision anyway
def exp(x: f24) -> f24:
  x2 = x * x
  x3 = x2 * x
  x4 = x3 * x
  x5 = x4 * x
  return 1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0 + x5 / 120.0

def fvec_exp(v: FVec) -> FVec:
  fold v:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      return FVec/Cons { head: exp(v.head), tail: v.tail }

def softmax(v: FVec) -> FVec:
  peak  = fvec_max(v, -999.0)
  use v = v
  exps  = fvec_exp(fvec_shift(v, peak))
  total = fvec_sum(exps)
  return fvec_scale(exps, 1.0 / total)

# bend is affine so vec gets used twice here (dot + recursion)
# the 'use' tells the compiler we know what we're doing and prevents nested dup bugs
def mat_vec_mul(mat: FMatrix, vec: FVec) -> FVec:
  fold mat with vec:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      return FVec/Cons { head: fvec_dot(mat.row, vec), tail: mat.tail(vec) }

def linear(mat: FMatrix, bias: FVec, vec: FVec) -> FVec:
  return fvec_add(mat_vec_mul(mat, vec), bias)

def score_query(query: FVec, keys: Seq, scale: f24) -> FVec:
  fold keys with query, scale:
    case Seq/Nil:
      return FVec/Nil
    case Seq/Cons:
      score = fvec_dot(query, keys.vec) * scale
      return FVec/Cons { head: score, tail: keys.tail(query, scale) }

def causal_mask(scores: FVec, key_pos: u24, query_pos: u24) -> FVec:
  fold scores with key_pos, query_pos:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      if key_pos > query_pos:
        return FVec/Cons { head: -999.0, tail: scores.tail(key_pos + 1, query_pos) }
      else:
        return FVec/Cons { head: scores.head, tail: scores.tail(key_pos + 1, query_pos) }

def weighted_sum(weights: FVec, values: Seq) -> FVec:
  fold weights with values:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      match values:
        case Seq/Nil:
          return FVec/Nil
        case Seq/Cons:
          scaled = fvec_scale(values.vec, weights.head)
          rest   = weights.tail(values.tail)
          return fvec_add(scaled, rest)

def attend(query: FVec, keys: Seq, values: Seq, pos: u24, scale: f24) -> FVec:
  scores  = score_query(query, keys, scale)
  masked  = causal_mask(scores, 0, pos)
  weights = softmax(masked)
  return weighted_sum(weights, values)

def ffn(input: FVec, w1: FMatrix, b1: FVec, w2: FMatrix, b2: FVec) -> FVec:
  hidden    = fvec_relu(linear(w1, b1, input))
  output    = linear(w2, b2, hidden)
  use input = input
  return fvec_add(input, output)

def project_seq(seq: Seq, mat: FMatrix) -> Seq:
  fold seq with mat:
    case Seq/Nil:
      return Seq/Nil
    case Seq/Cons:
      return Seq/Cons { vec: mat_vec_mul(mat, seq.vec), tail: seq.tail(mat) }

def attend_seq(hidden: Seq, keys: Seq, values: Seq, pos: u24, scale: f24) -> Seq:
  fold hidden with keys, values, pos, scale:
    case Seq/Nil:
      return Seq/Nil
    case Seq/Cons:
      attended = attend(hidden.vec, keys, values, pos, scale)
      residual = fvec_add(hidden.vec, attended)
      return Seq/Cons { vec: residual, tail: hidden.tail(keys, values, pos + 1, scale) }

def ffn_seq(seq: Seq, w1: FMatrix, b1: FVec, w2: FMatrix, b2: FVec) -> Seq:
  fold seq with w1, b1, w2, b2:
    case Seq/Nil:
      return Seq/Nil
    case Seq/Cons:
      return Seq/Cons {
        vec: ffn(seq.vec, w1, b1, w2, b2),
        tail: seq.tail(w1, b1, w2, b2)
      }

def seq_last(seq: Seq) -> FVec:
  match seq:
    case Seq/Nil:
      return FVec/Nil
    case Seq/Cons:
      match seq.tail:
        case Seq/Nil:
          return seq.vec
        case Seq/Cons:
          return seq_last(seq.tail)

def forward(tokens: Seq, weights: Weights) -> FVec:
  match weights:
    case Weights/Mk:
      scale     = 1.0 / 3.0 ** 0.5
      queries   = project_seq(tokens, weights.wq)
      keys      = project_seq(tokens, weights.wk)
      values    = project_seq(tokens, weights.wv)
      attended  = attend_seq(tokens, keys, values, 0, scale)
      projected = project_seq(attended, weights.wo)
      after_ffn = ffn_seq(projected, weights.w1, weights.b1, weights.w2, weights.b2)
      last      = seq_last(after_ffn)
      logits    = mat_vec_mul(weights.whead, last)
      return softmax(logits)

def argmax(probs: FVec, best_idx: u24, best_val: f24, cur_idx: u24) -> u24:
  fold probs with best_idx, best_val, cur_idx:
    case FVec/Nil:
      return best_idx
    case FVec/Cons:
      if probs.head > best_val:
        return probs.tail(cur_idx, probs.head, cur_idx + 1)
      else:
        return probs.tail(best_idx, best_val, cur_idx + 1)

def lookup(idx: u24, embeddings: Seq) -> FVec:
  fold embeddings with idx:
    case Seq/Nil:
      return FVec/Nil
    case Seq/Cons:
      if idx == 0:
        return embeddings.vec
      else:
        return embeddings.tail(idx - 1)

def seq_append(seq: Seq, vec: FVec) -> Seq:
  fold seq with vec:
    case Seq/Nil:
      return Seq/Cons { vec: vec, tail: Seq/Nil }
    case Seq/Cons:
      return Seq/Cons { vec: seq.vec, tail: seq.tail(vec) }

def generate(context: Seq, embeddings: Seq, weights: Weights, steps: u24) -> Tokens:
  if steps == 0:
    return Tokens/Nil
  else:
    use embeddings = embeddings
    use weights    = weights
    probs          = forward(context, weights)
    next_id        = argmax(probs, 0, -999.0, 0)
    next_vec       = lookup(next_id, embeddings)
    next_context   = seq_append(context, next_vec)
    rest           = generate(next_context, embeddings, weights, steps - 1)
    return Tokens/Cons { id: next_id, tail: rest }

# I needed to write FVec/Cons out because there's no array literal so guess who needs to spell it out node by node :)
def main() -> Tokens:
  t0 = FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  t1 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  t2 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Nil } } }
  zero = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }

  prompt = Seq/Cons { vec: t0, tail: Seq/Cons { vec: t1, tail: Seq/Nil } }

  use t0 = t0
  use t1 = t1
  use t2 = t2
  embeddings = Seq/Cons { vec: t0, tail: Seq/Cons { vec: t1, tail: Seq/Cons { vec: t2, tail: Seq/Nil } } }

  eye0 = FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  eye1 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  eye2 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Nil } } }

  use eye0 = eye0
  use eye1 = eye1
  use eye2 = eye2

  identity = FMatrix/Cons { row: eye0, tail: FMatrix/Cons { row: eye1, tail: FMatrix/Cons { row: eye2, tail: FMatrix/Nil } } }

  use identity = identity
  use identity = identity
  use identity = identity
  use identity = identity
  use identity = identity
  use identity = identity
  use zero     = zero

  weights = Weights/Mk {
    wq: identity, wk: identity, wv: identity, wo: identity,
    w1: identity, b1: zero,
    w2: identity, b2: zero,
    whead: identity
  }

  return generate(prompt, embeddings, weights, 3)
