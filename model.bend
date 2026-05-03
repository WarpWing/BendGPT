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

def fvec_dot(a: FVec, b: FVec) -> f24:
  match a:
    case FVec/Nil:
      return 0.0
    case FVec/Cons:
      match b:
        case FVec/Nil:
          return 0.0
        case FVec/Cons:
          return a.head * b.head + fvec_dot(a.tail, b.tail)

# treat Nil as a zero vector so weighted_sum bottoms out correctly
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
  match v:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      return FVec/Cons { head: v.head * s, tail: fvec_scale(v.tail, s) }

def fvec_relu(v: FVec) -> FVec:
  match v:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      if v.head > 0.0:
        return FVec/Cons { head: v.head, tail: fvec_relu(v.tail) }
      else:
        return FVec/Cons { head: 0.0, tail: fvec_relu(v.tail) }

def fvec_max(v: FVec, current: f24) -> f24:
  match v:
    case FVec/Nil:
      return current
    case FVec/Cons:
      if v.head > current:
        return fvec_max(v.tail, v.head)
      else:
        return fvec_max(v.tail, current)

def fvec_sum(v: FVec) -> f24:
  match v:
    case FVec/Nil:
      return 0.0
    case FVec/Cons:
      return v.head + fvec_sum(v.tail)

def fvec_shift(v: FVec, s: f24) -> FVec:
  match v:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      return FVec/Cons { head: v.head - s, tail: fvec_shift(v.tail, s) }

# bend has no exp() builtin so we use a taylor series
# 6 terms is enough given f24 only has ~3 decimal digits of precision anyway
def exp(x: f24) -> f24:
  x2 = x * x
  x3 = x2 * x
  x4 = x3 * x
  x5 = x4 * x
  return 1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0 + x5 / 120.0

def fvec_exp(v: FVec) -> FVec:
  match v:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      return FVec/Cons { head: exp(v.head), tail: fvec_exp(v.tail) }

def softmax(v: FVec) -> FVec:
  peak  = fvec_max(v, -999.0)
  use v = v
  exps  = fvec_exp(fvec_shift(v, peak))
  total = fvec_sum(exps)
  return fvec_scale(exps, 1.0 / total)

# bend is affine so vec gets used twice here (dot + recursion)
# the 'use' tells the compiler we know what we're doing and prevents nested dup bugs
def mat_vec_mul(mat: FMatrix, vec: FVec) -> FVec:
  match mat:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      use vec = vec
      return FVec/Cons { head: fvec_dot(mat.row, vec), tail: mat_vec_mul(mat.tail, vec) }

def linear(mat: FMatrix, bias: FVec, vec: FVec) -> FVec:
  return fvec_add(mat_vec_mul(mat, vec), bias)

def score_query(query: FVec, keys: Seq, scale: f24) -> FVec:
  match keys:
    case Seq/Nil:
      return FVec/Nil
    case Seq/Cons:
      use query = query
      score = fvec_dot(query, keys.vec) * scale
      return FVec/Cons { head: score, tail: score_query(query, keys.tail, scale) }

def causal_mask(scores: FVec, key_pos: u24, query_pos: u24) -> FVec:
  match scores:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      if key_pos > query_pos:
        return FVec/Cons { head: -999.0, tail: causal_mask(scores.tail, key_pos + 1, query_pos) }
      else:
        return FVec/Cons { head: scores.head, tail: causal_mask(scores.tail, key_pos + 1, query_pos) }

def weighted_sum(weights: FVec, values: Seq) -> FVec:
  match weights:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      match values:
        case Seq/Nil:
          return FVec/Nil
        case Seq/Cons:
          scaled = fvec_scale(values.vec, weights.head)
          rest   = weighted_sum(weights.tail, values.tail)
          return fvec_add(scaled, rest)

def attend(query: FVec, keys: Seq, values: Seq, pos: u24, scale: f24) -> FVec:
  scores  = score_query(query, keys, scale)
  masked  = causal_mask(scores, 0, pos)
  weights = softmax(masked)
  return weighted_sum(weights, values)

def ffn(input: FVec, w1: FMatrix, b1: FVec, w2: FMatrix, b2: FVec) -> FVec:
  hidden   = fvec_relu(linear(w1, b1, input))
  output   = linear(w2, b2, hidden)
  use input = input
  return fvec_add(input, output)

def ffn_seq(seq: Seq, w1: FMatrix, b1: FVec, w2: FMatrix, b2: FVec) -> Seq:
  match seq:
    case Seq/Nil:
      return Seq/Nil
    case Seq/Cons:
      use w1 = w1
      use b1 = b1
      use w2 = w2
      use b2 = b2
      return Seq/Cons {
        vec: ffn(seq.vec, w1, b1, w2, b2),
        tail: ffn_seq(seq.tail, w1, b1, w2, b2)
      }

# I needed to write FVec/Cons out because there's no array literal so guess who needs to spell it out node by node :)
def main() -> Seq:

  t0 = FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  t1 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  t2 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Nil } } }

  seq = Seq/Cons { vec: t0, tail: Seq/Cons { vec: t1, tail: Seq/Cons { vec: t2, tail: Seq/Nil } } }

  row0 = FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  row1 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  row2 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Nil } } }

  use row0 = row0
  use row1 = row1
  use row2 = row2

  w1  = FMatrix/Cons { row: row0, tail: FMatrix/Cons { row: row1, tail: FMatrix/Cons { row: row2, tail: FMatrix/Nil } } }
  w2  = FMatrix/Cons { row: row0, tail: FMatrix/Cons { row: row1, tail: FMatrix/Cons { row: row2, tail: FMatrix/Nil } } }
  b1  = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  b2  = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }

  return ffn_seq(seq, w1, b1, w2, b2)
