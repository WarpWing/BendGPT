import struct, random, math, sys

weights_file = sys.argv[1] if len(sys.argv) > 1 else "weights.bin"
out_file     = sys.argv[2] if len(sys.argv) > 2 else "model.bend"

with open(weights_file, "rb") as f:
    def read_floats(f, n):
        return list(struct.unpack(f"{n}f", f.read(4 * n)))

    vocab_size = struct.unpack("I", f.read(4))[0]
    chars      = [chr(struct.unpack("B", f.read(1))[0]) for _ in range(vocab_size)]
    n_layers   = struct.unpack("I", f.read(4))[0]
    embed      = struct.unpack("I", f.read(4))[0]
    hidden     = struct.unpack("I", f.read(4))[0]
    ctx        = struct.unpack("I", f.read(4))[0]

    token_emb = [read_floats(f, embed) for _ in range(vocab_size)]
    pos_emb   = [read_floats(f, embed) for _ in range(ctx)]

    blocks = []
    for _ in range(n_layers):
        rms1 = read_floats(f, embed)
        wq   = [read_floats(f, embed) for _ in range(embed)]
        wk   = [read_floats(f, embed) for _ in range(embed)]
        wv   = [read_floats(f, embed) for _ in range(embed)]
        wo   = [read_floats(f, embed) for _ in range(embed)]
        rms2 = read_floats(f, embed)
        w1   = [read_floats(f, embed) for _ in range(hidden)]
        b1   = read_floats(f, hidden)
        w2   = [read_floats(f, hidden) for _ in range(embed)]
        b2   = read_floats(f, embed)
        blocks.append((rms1, wq, wk, wv, wo, rms2, w1, b1, w2, b2))

    rms_final = read_floats(f, embed)
    lm_head   = [read_floats(f, embed) for _ in range(vocab_size)]

attn_scale = 1.0 / math.sqrt(embed)
rms_div    = float(embed)
ctx_max    = ctx - 1

def gen_direct_lookup(fn_name, row_prefix, n):
    def gen(lo, hi, depth):
        ind = "  " * depth
        if lo == hi:
            return f"{ind}return {row_prefix}_row_{lo}()"
        mid = (lo + hi) // 2
        left = gen(lo, mid, depth + 1)
        right = gen(mid + 1, hi, depth + 1)
        return "\n".join([
            f"{ind}if idx <= {mid}:",
            left if lo == mid else left,
            f"{ind}else:",
            right,
        ])
    return f"def {fn_name}(idx: u24) -> FVec:\n{gen(0, n - 1, 1)}"

def gen_blk_steps(n, attn_scale):
    lines = []
    for i in range(n):
        lines += [
            f"def blk{i}_step(x: FVec, k_cache: Seq, v_cache: Seq) -> BlkResult:",
            f"  nx     = rmsnorm(x, get_blk{i}_rms1())",
            f"  use nx = nx",
            f"  k_new  = mat_vec_mul(get_blk{i}_wk(), nx)",
            f"  v_new  = mat_vec_mul(get_blk{i}_wv(), nx)",
            f"  k      = seq_prepend(k_cache, k_new)",
            f"  v      = seq_prepend(v_cache, v_new)",
            f"  use k  = k",
            f"  use v  = v",
            f"  q      = mat_vec_mul(get_blk{i}_wq(), nx)",
            f"  attn   = attend_last(q, k, v, {attn_scale:.6f})",
            f"  xa     = add_matvec_fold(x, get_blk{i}_wo(), attn)",
            f"  use xa = xa",
            f"  nxb    = rmsnorm(xa, get_blk{i}_rms2())",
            f"  h      = linear_gelu_fold(get_blk{i}_w1(), get_blk{i}_b1(), nxb)",
            f"  out    = add_linear_fold(xa, get_blk{i}_w2(), get_blk{i}_b2(), h)",
            f"  return BlkResult/Mk {{ out: out, keys: k, vals: v }}",
            "",
        ]
    return "\n".join(lines)

def gen_forward_cached(n):
    lines = [
        "def forward_cached(x: FVec, cache: GenCache) -> StepResult:",
        "  match cache:",
        "    case GenCache/Mk:",
    ]
    for i in range(n):
        ind  = "      " + "    " * i
        prev = "x" if i == 0 else f"r{i-1}.out"
        lines.append(f"{ind}r{i} = blk{i}_step({prev}, cache.k{i}, cache.v{i})")
        lines.append(f"{ind}match r{i}:")
        lines.append(f"{ind}  case BlkResult/Mk:")
    ind = "      " + "    " * n
    cache_mk = ", ".join(f"k{i}: r{i}.keys, v{i}: r{i}.vals" for i in range(n))
    lines += [
        f"{ind}normed    = rmsnorm(r{n-1}.out, get_rms_final())",
        f"{ind}logits    = mat_vec_mul(get_lm_head(), normed)",
        f"{ind}probs     = softmax(fvec_scale(logits, 0.77))",
        f"{ind}new_cache = GenCache/Mk {{ {cache_mk} }}",
        f"{ind}return StepResult/Mk {{ probs: probs, cache: new_cache }}",
    ]
    return "\n".join(lines)

cache_fields = "\n    ".join(
    f"k{i}: Seq, v{i}: Seq" + ("," if i < n_layers - 1 else "")
    for i in range(n_layers)
)
empty_cache_fields = ", ".join(f"k{i}: Seq/Nil, v{i}: Seq/Nil" for i in range(n_layers))

BASE_CODE = f"""type FVec:
  Nil
  Cons {{ head: f24, ~tail: FVec }}

type FMatrix:
  Nil
  Cons {{ row: FVec, ~tail: FMatrix }}

type Seq:
  Nil
  Cons {{ vec: FVec, ~tail: Seq }}

type Tokens:
  Nil
  Cons {{ id: u24, ~tail: Tokens }}

type GenCache:
  Mk {{
    {cache_fields}
  }}

type BlkResult:
  Mk {{ out: FVec, keys: Seq, vals: Seq }}

type StepResult:
  Mk {{ probs: FVec, cache: GenCache }}

type ExpSum:
  Mk {{ exps: FVec, total: f24 }}

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
          return FVec/Cons {{ head: a.head + b.head, tail: fvec_add(a.tail, b.tail) }}

def fvec_scale(v: FVec, s: f24) -> FVec:
  fold v with s:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      return FVec/Cons {{ head: v.head * s, tail: v.tail(s) }}

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
          return FVec/Cons {{ head: v.head * inv * scale.head, tail: v.tail(scale.tail, inv) }}

def rmsnorm(v: FVec, scale: FVec) -> FVec:
  sq  = fvec_sumsq(v)
  use v = v
  inv = 1.0 / ((sq / {rms_div} + 0.00001) ** 0.5)
  return rmsnorm_vec(v, scale, inv)

def mat_vec_mul(mat: FMatrix, vec: FVec) -> FVec:
  fold mat with vec:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      return FVec/Cons {{ head: fvec_dot(mat.row, vec), tail: mat.tail(vec) }}

def linear_gelu_fold(mat: FMatrix, bias: FVec, vec: FVec) -> FVec:
  fold mat with vec, bias:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      match bias:
        case FVec/Nil:
          return FVec/Nil
        case FVec/Cons:
          return FVec/Cons {{ head: gelu(fvec_dot(mat.row, vec) + bias.head), tail: mat.tail(vec, bias.tail) }}

def add_matvec_fold(residual: FVec, mat: FMatrix, vec: FVec) -> FVec:
  fold mat with vec, residual:
    case FMatrix/Nil:
      return FVec/Nil
    case FMatrix/Cons:
      match residual:
        case FVec/Nil:
          return FVec/Nil
        case FVec/Cons:
          return FVec/Cons {{ head: fvec_dot(mat.row, vec) + residual.head, tail: mat.tail(vec, residual.tail) }}

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
              return FVec/Cons {{ head: fvec_dot(mat.row, vec) + bias.head + residual.head, tail: mat.tail(vec, bias.tail, residual.tail) }}

def fvec_exp_sum(v: FVec, shift: f24) -> ExpSum:
  fold v with shift:
    case FVec/Nil:
      return ExpSum/Mk {{ exps: FVec/Nil, total: 0.0 }}
    case FVec/Cons:
      e    = exp(v.head - shift)
      rest = v.tail(shift)
      match rest:
        case ExpSum/Mk:
          return ExpSum/Mk {{ exps: FVec/Cons {{ head: e, tail: rest.exps }}, total: e + rest.total }}

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
      return FVec/Cons {{ head: score, tail: keys.tail(query, scale) }}

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
  return Seq/Cons {{ vec: vec, tail: seq }}

def threshold_filter(probs: FVec, thresh: f24) -> FVec:
  fold probs with thresh:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      if probs.head < thresh:
        return FVec/Cons {{ head: 0.0, tail: probs.tail(thresh) }}
      else:
        return FVec/Cons {{ head: probs.head, tail: probs.tail(thresh) }}

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
      return Seq/Cons {{ vec: vec, tail: encode_prompt(prompt.tail, vocab, pos + 1) }}

def tokens_to_string(tokens: Tokens, vocab: List(u24)) -> String:
  match tokens:
    case Tokens/Nil:
      return String/Nil
    case Tokens/Cons:
      use vocab = vocab
      ch = idx_to_char(tokens.id, vocab, 0)
      return String/Cons {{ head: ch, tail: tokens_to_string(tokens.tail, vocab) }}

def string_append(a: String, b: String) -> String:
  match a:
    case String/Nil:
      return b
    case String/Cons:
      return String/Cons {{ head: a.head, tail: string_append(a.tail, b) }}

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
  if n > {ctx_max}:
    return {ctx_max}
  else:
    return n

def empty_cache() -> GenCache:
  return GenCache/Mk {{ {empty_cache_fields} }}

{gen_blk_steps(n_layers, attn_scale)}

{gen_forward_cached(n_layers)}

def init_cache_loop(prompt: Seq, cache: GenCache, last_probs: FVec) -> StepResult:
  match prompt:
    case Seq/Nil:
      return StepResult/Mk {{ probs: last_probs, cache: cache }}
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
        return Tokens/Cons {{ id: next_id, tail: rest }}
"""

def fvec(row):
    s = "FVec/Nil"
    for v in reversed(row):
        s = f"FVec/Cons {{ head: {v:.6f}, tail: {s} }}"
    return s

def bend_char(c):
    if c == "'": return "'\\''"
    if c == '\\': return "'\\\\'"
    return f"'{c}'"

def emit_matrix(lines, name, rows, assemble=True):
    for i, row in enumerate(rows):
        lines.append(f"def {name}_row_{i}() -> FVec:")
        lines.append(f"  return {fvec(row)}")
        lines.append("")
    if assemble:
        body = "FMatrix/Nil"
        for i in reversed(range(len(rows))):
            body = f"FMatrix/Cons {{ row: {name}_row_{i}(), tail: {body} }}"
        lines.append(f"def {name}() -> FMatrix:")
        lines.append(f"  return {body}")
        lines.append("")

def emit_vec(lines, name, row):
    lines.append(f"def {name}() -> FVec:")
    lines.append(f"  return {fvec(row)}")
    lines.append("")

lines = [BASE_CODE]

char_list = "List/Nil"
for c in reversed(chars):
    char_list = f"List/Cons {{ head: {bend_char(c)}, tail: {char_list} }}"
lines.append("def get_chars() -> List(u24):")
lines.append(f"  return {char_list}")
lines.append("")

emit_matrix(lines, "get_token_emb", token_emb, assemble=False)
emit_matrix(lines, "get_pos_emb",   pos_emb,   assemble=False)
lines.append(gen_direct_lookup("get_token", "get_token_emb", vocab_size))
lines.append("")
lines.append(gen_direct_lookup("get_pos", "get_pos_emb", ctx))
lines.append("")

for i, (rms1, wq, wk, wv, wo, rms2, w1, b1, w2, b2) in enumerate(blocks):
    emit_vec(lines,    f"get_blk{i}_rms1", rms1)
    emit_matrix(lines, f"get_blk{i}_wq",   wq)
    emit_matrix(lines, f"get_blk{i}_wk",   wk)
    emit_matrix(lines, f"get_blk{i}_wv",   wv)
    emit_matrix(lines, f"get_blk{i}_wo",   wo)
    emit_vec(lines,    f"get_blk{i}_rms2", rms2)
    emit_matrix(lines, f"get_blk{i}_w1",   w1)
    emit_vec(lines,    f"get_blk{i}_b1",   b1)
    emit_matrix(lines, f"get_blk{i}_w2",   w2)
    emit_vec(lines,    f"get_blk{i}_b2",   b2)

emit_vec(lines,    "get_rms_final", rms_final)
emit_matrix(lines, "get_lm_head",   lm_head)

N_GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 48

rng_vals  = [random.random() for _ in range(N_GEN)]
rand_list = "List/Nil"
for v in reversed(rng_vals):
    rand_list = f"List/Cons {{ head: {v:.6f}, tail: {rand_list} }}"
lines.append("def get_randoms() -> List(f24):")
lines.append(f"  return {rand_list}")
lines.append("")

prompt     = "ROMEO:\nI "
prompt_len = len(prompt)

lines.append("def main() -> String:")
lines.append("  chars   = get_chars()")
lines.append("  use chars = chars")
lines.append("  randoms = get_randoms()")
lines.append(f'  prompt  = encode_prompt("ROMEO:\\nI ", chars, 0)')
lines.append("  init    = init_cache_loop(prompt, empty_cache(), FVec/Nil)")
lines.append("  match init:")
lines.append("    case StepResult/Mk:")
lines.append(f"      result = generate_kv(init.probs, init.cache, {N_GEN}, randoms, {prompt_len})")
lines.append("      output = tokens_to_string(result, chars)")
lines.append('      return string_append("ROMEO:\\nI ", string_append(output, "\\n"))')

with open(out_file, "w") as f:
    f.write("\n".join(lines))

print(f"done — vocab={vocab_size}, embed={embed}, hidden={hidden}, layers={n_layers}, ctx={ctx}")
print(f"written: {out_file}")
