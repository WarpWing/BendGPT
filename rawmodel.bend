# The bend model without the weights baked in
type FVec:
  Nil
  Cons { head: f24, ~tail: FVec }

type FMatrix:
  Nil
  Cons { row: FVec, ~tail: FMatrix }

type Seq:
  Nil
  Cons { vec: FVec, ~tail: Seq }

type Tokens:
  Nil
  Cons { id: u24, ~tail: Tokens }

type GenCache:
  Mk {
    k0: Seq, v0: Seq,
    k1: Seq, v1: Seq
  }

type BlkResult:
  Mk { out: FVec, keys: Seq, vals: Seq }

type StepResult:
  Mk { probs: FVec, cache: GenCache }

type ExpSum:
  Mk { exps: FVec, total: f24 }


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

def fvec_sumsq(v: FVec) -> f24:
  fold v:
    case FVec/Nil:
      return 0.0
    case FVec/Cons:
      return v.head * v.head + v.tail


def pade_unit(y: f24) -> f24:
  y2  = y * y
  num = 1.0 + y / 2.0 + y2 / 12.0
  den = 1.0 - y / 2.0 + y2 / 12.0
  return num / den

def exp(x: f24) -> f24:
  e  = pade_unit(x / 4.0)
  e2 = e * e
  return e2 * e2


def gelu(x: f24) -> f24:
  return x * (1.0 / (1.0 + exp(0.0 - 1.702 * x)))


def rmsnorm_vec(v: FVec, scale: FVec, inv: f24) -> FVec:
  fold v with scale, inv:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      match scale:
        case FVec/Nil:
          return FVec/Nil
        case FVec/Cons:
          return FVec/Cons { head: v.head * inv * scale.head, tail: v.tail(scale.tail, inv) }


def rmsnorm(v: FVec, scale: FVec) -> FVec:
  sq  = fvec_sumsq(v)
  use v = v
  inv = 1.0 / ((sq / 48.0 + 0.00001) ** 0.5)
  return rmsnorm_vec(v, scale, inv)


def mat_vec_mul(mat: FMatrix, vec: FVec) -> FVec:
  fold mat with vec:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      return FVec/Cons { head: fvec_dot(mat.row, vec), tail: mat.tail(vec) }


def linear_gelu_fold(mat: FMatrix, bias: FVec, vec: FVec) -> FVec:
  fold mat with vec, bias:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      match bias:
        case FVec/Nil:
          return FVec/Nil
        case FVec/Cons:
          return FVec/Cons { head: gelu(fvec_dot(mat.row, vec) + bias.head), tail: mat.tail(vec, bias.tail) }


def add_matvec_fold(residual: FVec, mat: FMatrix, vec: FVec) -> FVec:
  fold mat with vec, residual:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      match residual:
        case FVec/Nil:
          return FVec/Nil
        case FVec/Cons:
          return FVec/Cons { head: fvec_dot(mat.row, vec) + residual.head, tail: mat.tail(vec, residual.tail) }


def add_linear_fold(residual: FVec, mat: FMatrix, bias: FVec, vec: FVec) -> FVec:
  fold mat with vec, bias, residual:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      match bias:
        case FVec/Nil:
          return FVec/Nil
        case FVec/Cons:
          match residual:
            case FVec/Nil:
              return FVec/Nil
            case FVec/Cons:
              return FVec/Cons { head: fvec_dot(mat.row, vec) + bias.head + residual.head, tail: mat.tail(vec, bias.tail, residual.tail) }


def fvec_exp_sum(v: FVec, shift: f24) -> ExpSum:
  fold v with shift:
    case FVec/Nil:
      return ExpSum/Mk { exps: FVec/Nil, total: 0.0 }
    case FVec/Cons:
      e    = exp(v.head - shift)
      rest = v.tail(shift)
      match rest:
        case ExpSum/Mk:
          return ExpSum/Mk { exps: FVec/Cons { head: e, tail: rest.exps }, total: e + rest.total }

def softmax(v: FVec) -> FVec:
  peak = fvec_max(v, -999.0)
  use v = v
  er   = fvec_exp_sum(v, peak)
  match er:
    case ExpSum/Mk:
      return fvec_scale(er.exps, 1.0 / er.total)


def score_query(query: FVec, keys: Seq, scale: f24) -> FVec:
  fold keys with query, scale:
    case Seq/Nil:
      return FVec/Nil
    case Seq/Cons:
      score = fvec_dot(query, keys.vec) * scale
      return FVec/Cons { head: score, tail: keys.tail(query, scale) }


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


def attend_last(query: FVec, keys: Seq, values: Seq, scale: f24) -> FVec:
  scores  = score_query(query, keys, scale)
  weights = softmax(scores)
  return weighted_sum(weights, values)


def seq_prepend(seq: Seq, vec: FVec) -> Seq:
  return Seq/Cons { vec: vec, tail: seq }


def threshold_filter(probs: FVec, thresh: f24) -> FVec:
  fold probs with thresh:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      if probs.head < thresh:
        return FVec/Cons { head: 0.0, tail: probs.tail(thresh) }
      else:
        return FVec/Cons { head: probs.head, tail: probs.tail(thresh) }


def sample_cdf(probs: FVec, r: f24, idx: u24) -> u24:
  fold probs with r, idx:
    case FVec/Nil:
      return idx
    case FVec/Cons:
      if r < probs.head:
        return idx
      else:
        return probs.tail(r - probs.head, idx + 1)

def filtered_sample(probs: FVec, thresh: f24, r: f24) -> u24:
  filtered = threshold_filter(probs, thresh)
  total    = fvec_sum(filtered)
  normed   = fvec_scale(filtered, 1.0 / total)
  return sample_cdf(normed, r, 0)


def char_to_idx(ch: u24, vocab: List(u24), idx: u24) -> u24:
  match vocab:
    case List/Nil:
      return 0
    case List/Cons:
      if vocab.head == ch:
        return idx
      else:
        return char_to_idx(ch, vocab.tail, idx + 1)

def idx_to_char(idx: u24, vocab: List(u24), cur: u24) -> u24:
  match vocab:
    case List/Nil:
      return '?'
    case List/Cons:
      if cur == idx:
        return vocab.head
      else:
        return idx_to_char(idx, vocab.tail, cur + 1)

def encode_prompt(prompt: String, vocab: List(u24), pos: u24) -> Seq:
  match prompt:
    case String/Nil:
      return Seq/Nil
    case String/Cons:
      use vocab = vocab
      idx  = char_to_idx(prompt.head, vocab, 0)
      tvec = get_token(idx)
      pvec = get_pos(pos)
      vec  = fvec_add(tvec, pvec)
      return Seq/Cons { vec: vec, tail: encode_prompt(prompt.tail, vocab, pos + 1) }

def tokens_to_string(tokens: Tokens, vocab: List(u24)) -> String:
  match tokens:
    case Tokens/Nil:
      return String/Nil
    case Tokens/Cons:
      use vocab = vocab
      ch = idx_to_char(tokens.id, vocab, 0)
      return String/Cons { head: ch, tail: tokens_to_string(tokens.tail, vocab) }

def string_append(a: String, b: String) -> String:
  match a:
    case String/Nil:
      return b
    case String/Cons:
      return String/Cons { head: a.head, tail: string_append(a.tail, b) }


def get_r(randoms: List(f24)) -> f24:
  match randoms:
    case List/Nil:
      return 0.5
    case List/Cons:
      return randoms.head

def advance_randoms(randoms: List(f24)) -> List(f24):
  match randoms:
    case List/Nil:
      return List/Nil
    case List/Cons:
      return randoms.tail


def pos_clamp(n: u24) -> u24:
  if n > 63:
    return 63
  else:
    return n


def empty_cache() -> GenCache:
  return GenCache/Mk { k0: Seq/Nil, v0: Seq/Nil, k1: Seq/Nil, v1: Seq/Nil }


def blk0_step(x: FVec, k_cache: Seq, v_cache: Seq) -> BlkResult:
  nx     = rmsnorm(x, get_blk0_rms1())
  use nx = nx
  k_new  = mat_vec_mul(get_blk0_wk(), nx)
  v_new  = mat_vec_mul(get_blk0_wv(), nx)
  k      = seq_prepend(k_cache, k_new)
  v      = seq_prepend(v_cache, v_new)
  use k  = k
  use v  = v
  q      = mat_vec_mul(get_blk0_wq(), nx)
  attn   = attend_last(q, k, v, 0.144338)
  xa     = add_matvec_fold(x, get_blk0_wo(), attn)
  use xa = xa
  nxb    = rmsnorm(xa, get_blk0_rms2())
  h      = linear_gelu_fold(get_blk0_w1(), get_blk0_b1(), nxb)
  out    = add_linear_fold(xa, get_blk0_w2(), get_blk0_b2(), h)
  return BlkResult/Mk { out: out, keys: k, vals: v }

def blk1_step(x: FVec, k_cache: Seq, v_cache: Seq) -> BlkResult:
  nx     = rmsnorm(x, get_blk1_rms1())
  use nx = nx
  k_new  = mat_vec_mul(get_blk1_wk(), nx)
  v_new  = mat_vec_mul(get_blk1_wv(), nx)
  k      = seq_prepend(k_cache, k_new)
  v      = seq_prepend(v_cache, v_new)
  use k  = k
  use v  = v
  q      = mat_vec_mul(get_blk1_wq(), nx)
  attn   = attend_last(q, k, v, 0.144338)
  xa     = add_matvec_fold(x, get_blk1_wo(), attn)
  use xa = xa
  nxb    = rmsnorm(xa, get_blk1_rms2())
  h      = linear_gelu_fold(get_blk1_w1(), get_blk1_b1(), nxb)
  out    = add_linear_fold(xa, get_blk1_w2(), get_blk1_b2(), h)
  return BlkResult/Mk { out: out, keys: k, vals: v }


def forward_cached(x: FVec, cache: GenCache) -> StepResult:
  match cache:
    case GenCache/Mk:
      r0 = blk0_step(x, cache.k0, cache.v0)
      match r0:
        case BlkResult/Mk:
          r1 = blk1_step(r0.out, cache.k1, cache.v1)
          match r1:
            case BlkResult/Mk:
              normed    = rmsnorm(r1.out, get_rms_final())
              logits    = mat_vec_mul(get_lm_head(), normed)
              probs     = softmax(fvec_scale(logits, 0.77))
              new_cache = GenCache/Mk { k0: r0.keys, v0: r0.vals, k1: r1.keys, v1: r1.vals }
              return StepResult/Mk { probs: probs, cache: new_cache }


def init_cache_loop(prompt: Seq, cache: GenCache, last_probs: FVec) -> StepResult:
  match prompt:
    case Seq/Nil:
      return StepResult/Mk { probs: last_probs, cache: cache }
    case Seq/Cons:
      result = forward_cached(prompt.vec, cache)
      match result:
        case StepResult/Mk:
          return init_cache_loop(prompt.tail, result.cache, result.probs)


def generate_kv(probs: FVec, cache: GenCache, steps: u24, randoms: List(f24), ctx_pos: u24) -> Tokens:
  if steps == 0:
    return Tokens/Nil
  else:
    use randoms = randoms
    r        = get_r(randoms)
    next_id  = filtered_sample(probs, 0.05, r)
    raw_vec  = get_token(next_id)
    pvec     = get_pos(pos_clamp(ctx_pos))
    input    = fvec_add(raw_vec, pvec)
    result   = forward_cached(input, cache)
    match result:
      case StepResult/Mk:
        rest = generate_kv(result.probs, result.cache, steps - 1, advance_randoms(randoms), ctx_pos + 1)
        return Tokens/Cons { id: next_id, tail: rest }


def get_token_stub() -> FVec:
  return FVec/Nil


def get_pos_stub() -> FVec:
  return FVec/Nil


def get_blk0_rms1() -> FVec:
  return FVec/Nil


def get_blk0_wq() -> FMatrix:
  return FMatrix/Nil


def get_blk0_wk() -> FMatrix:
  return FMatrix/Nil


def get_blk0_wv() -> FMatrix:
  return FMatrix/Nil


def get_blk0_wo() -> FMatrix:
  return FMatrix/Nil


def get_blk0_rms2() -> FVec:
  return FVec/Nil


def get_blk0_w1() -> FMatrix:
  return FMatrix/Nil


def get_blk0_b1() -> FVec:
  return FVec/Nil


def get_blk0_w2() -> FMatrix:
  return FMatrix/Nil


def get_blk0_b2() -> FVec:
  return FVec/Nil


def get_blk1_rms1() -> FVec:
  return FVec/Nil


def get_blk1_wq() -> FMatrix:
  return FMatrix/Nil


def get_blk1_wk() -> FMatrix:
  return FMatrix/Nil


def get_blk1_wv() -> FMatrix:
  return FMatrix/Nil


def get_blk1_wo() -> FMatrix:
  return FMatrix/Nil


def get_blk1_rms2() -> FVec:
  return FVec/Nil


def get_blk1_w1() -> FMatrix:
  return FMatrix/Nil


def get_blk1_b1() -> FVec:
  return FVec/Nil


def get_blk1_w2() -> FMatrix:
  return FMatrix/Nil


def get_blk1_b2() -> FVec:
  return FVec/Nil


def get_rms_final() -> FVec:
  return FVec/Nil


def get_lm_head() -> FMatrix:
  return FMatrix/Nil


def main() -> String:
  chars   = get_chars()
  use chars = chars
  randoms = get_randoms()
  prompt  = encode_prompt("ROMEO:\nI ", chars, 0)
  init    = init_cache_loop(prompt, empty_cache(), FVec/Nil)
  match init:
    case StepResult/Mk:
      result = generate_kv(init.probs, init.cache, 48, randoms, 9)
      output = tokens_to_string(result, chars)
      return string_append("ROMEO:\nI ", string_append(output, "\n"))
