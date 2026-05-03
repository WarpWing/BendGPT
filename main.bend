# bend only has 24-bit floats (f24) so I can't be as precise as I want :(
type FVec:
  Nil
  Cons { head: f24, ~tail: FVec }

type FMatrix:
  Nil
  Cons { row: FVec, ~tail: FMatrix }

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

def fvec_add(a: FVec, b: FVec) -> FVec:
  match a:
    case FVec/Nil:
      return FVec/Nil
    case FVec/Cons:
      match b:
        case FVec/Nil:
          return FVec/Nil
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

# I needed to write FVec/Cons out because there's no array literal so guess who needs to spell it out node by node :)
def main() -> FVec:
  row0 = FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  row1 = FVec/Cons { head: 0.0, tail: FVec/Cons { head: 1.0, tail: FVec/Cons { head: 0.0, tail: FVec/Nil } } }
  mat  = FMatrix/Cons { row: row0, tail: FMatrix/Cons { row: row1, tail: FMatrix/Nil } }
  # input vector [3.0, 4.0, 5.0]
  vec  = FVec/Cons { head: 3.0, tail: FVec/Cons { head: 4.0, tail: FVec/Cons { head: 5.0, tail: FVec/Nil } } }
  # should be [3.0, 4.0]
  return mat_vec_mul(mat, vec)
