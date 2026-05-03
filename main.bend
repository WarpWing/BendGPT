# bend only has 24-bit floats (f24) so I can't be as precise as I want :(
type FVec:
  Nil
  Cons { head: f24, ~tail: FVec }

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

# This should return 32.0
def main() -> f24:
  a = FVec/Cons { head: 1.0, tail: FVec/Cons { head: 2.0, tail: FVec/Cons { head: 3.0, tail: FVec/Nil } } }
  b = FVec/Cons { head: 4.0, tail: FVec/Cons { head: 5.0, tail: FVec/Cons { head: 6.0, tail: FVec/Nil } } }
  return fvec_dot(a, b) 
