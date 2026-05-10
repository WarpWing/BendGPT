# BendGPT
A mock transformer language model written in [Bend](https://github.com/HigherOrderCO/Bend).

## what is & why are we using bend

Bend is a programming language that looks like Python but runs in parallel on CPUs and GPUs automatically. You write your functions normally and it figures out what can run at the same time (don't ask me how, interaction net wizardry), which turns out to matter a lot for language models.

When a transformer processes text, most of the work can happen simultaneously. Tokens, attention heads, individual calculations are all independent and do not need to wait for each other. The hard part has always been telling the hardware to run things in parallel. That requires low-level, manually written GPU code, which is complex and time-consuming to get right. Bend does it automatically. You get the speed without writing a single line of parallel code.

This project aims to experiment and understand what it looks like to build a GPT-style character level transformer in Bend. The benefit is seeing how much Bend's automatic parallelization can accelerate model architecture without writing a single line of thread management code.

## what happened

Truthfully, I had a blast writing this project, but it was way harder than expected and I didn't have high hopes for this project.
Mostly because I knew Bend supported only 24-bit floats and the precision was going to be way off but I assume at smaller scales the output would be somewhat ok. 
But also, I knew that not everything would benefit from Bend but I reckoned if I came out learning the language (and some of it's tradeoffs and quirks) that I would consider this time not wasted (and in this regard, I succeeded in learning some interesting things).

### a quick rundown about HVM2

Firstly, I took some time to understand how HVM actually works. 

Most runtimes execute programs as a sequence of instructions operating on memory. You have variables, a call stack, and the CPU marches through instructions one by one, reading and writing to addresses. HVM2 works entirely differently.

HVM2 works as an interaction net evaluator. It's moreso a graph rewriting system where the entire program is represented as a nodes. Computation, in this regard, happens when two nodes meet at their principal ports (their primary connection points), forming what is called a redex. When a redex is found, the two nodes annihilate and are replaced by new nodes according to fixed rewrite rules. There are no mutable variables. There is no instruction pointer. There is only the graph, and the rules for rewriting it.

The reason why this matters (for performance reasons) is because in a conventional runtime, let's say an array of 48 floats is 48 * 4 = 192 contiguos bytes of memory in place. 
A dot product over that array is vectorizable (most modern CPUs can multiply and add 16 floats using SIMD) and the entire array fits in a single cache line. 
However, HVM has no such equivalent as a contiguous array (aka no array literals). Therefore, a list of 48 floats became 48 separate heap-allocated `Cons` nodes, each one pointing to the next by a pointer.
Now this means a dot product in this case fires 48 sequential graph rewrites, which touches 48 scattered memory locations with 1. no cache locality and 2. no opportunity for vectorization. This is IMPORTANT as this is the foundational why everything kinda sucked. 

### first model

So, since Bend has no array literals, every weight matrix had to be encoded as a nested linked list of constructors. I used a 256x256 weight matrix which became  a chain of 256 `FMatrix/Cons` nodes, each holding a chain of 256 `FVec/Cons` nodes for the row. The resulting `.bend` source file was **117MB**.

Now when you run `bend run-c` starts, it compiles the `.bend` source into a book of interaction net definitions, it allocates a fixed (and YES, a fixed) 4GB node heap before executing a single reduction. This is because Bend is a 32-bit architecture and so after 4GB, there's considerable peformance drops. 
This also mean any transformer forward pass pushed the total (loading and representing took 2.7GB of that heap. I forgot to mention since Bend has very limited I/O, I had to write a convert file that hardcoded the weights into the files. Incredibly bad but for a PoC, you do what you can do) pushed everything over the 4GB limit.
Note that this was just for running the C backend.The `run-rs` backend (Rust) has no such cap and grows its heap dynamically, which is how the original model eventually ran to completion but at **39 minutes* for 48 tokens. 

### dup problems

However, even before the memory wall, there was an even greater problem with how interaction nets handle share data. In a conventiona language, passing a variable to a recursive function is free. As such, the generation loop looked like this:

```python
def generate_kv(weights, cache, ctx, steps):
  use weights = weights
  ...
  generate_kv(weights, new_cache, new_ctx, steps - 1)
```

The `use weights = weights` line looks like a harmless local alias. In HVM, it is an explicit instruction to eagerly duplicate the `weights` structure so both the current step and the recursive call have their own copy. 
For N generation steps, HVM attempted to clone the entire weight tree N times simultaneously before beginning any actual computation because the recursive structure of the interaction net unrolled the entire duplication chain at graph-construction time. 
Essentialy, that meant trying to hold 48 independent copies of 12.8MB of weights in memory at once.

As such, I made the decision to elimate the entire weight parameter from the call chain and instead, call a weight getter function for each block step. With no shared reference, HVM has nothing to duplicate.

### kv cache clutch

In PyTorch, KV-caching is an inference optimization as it avoids recomputing key and value projections for tokens already in context, cutting attention from O(T²) to O(T).

However, KV caching more or less saved this project. Without caching, each generation step recomputed the full attention context from scratch over the entire sequence. In interaction net terms, the graph for step T contained the graph for step T-1 as a subgraph, which contained T-2, and so on. This means a quadratic node count to the point HVM's reduction engine couldn't make progress. 
With KV-caching, each step appends exactly one new K vector and one new V vector to a growing cache. The interaction net graph stays linear in the number of steps.

I also found a neat little micro-optimization in that attention over a K/V cache is order-independent and that the softmax-weighted sum over V produces the same result whether keys are stored newest-first or oldest-first.
So I replaced `seq_append` with `seq_prepend`, which adds to the head in O(1). For 48 generation steps over 2 layers, this eliminated roughly 2,400 unnecessary list traversals across the full generation run.

### the final numbers

| Backend | Time (48 tokens) |
|---|---|
| Python NumPy | 0.11s |
| Bend run-c (before optimizations) | 13.8s |
| Bend run-c (after optimizations) | 13.1s |
| Bend run-rs (before optimizations) | 17.9s |
| Bend run-rs (after optimizations) | 15.1s |

# a short note about cuda + conclusion 

So Bend has an experimental GPU backend (`bend run-cu`). It maps interaction net reductions onto CUDA threads so many independent redexes can fire in parallel. 

CUDA warps one instruction over 32 uniform threads (classic SIMD right). What I learned is that HVM's GPU backend instead fires many independent graph rewrites in parallel, which is powerful when you have millions of structurally independent redexes: symbolic evaluation, combinator calculus, parallel tree search (this gave me an idea for a chess based project with Bend, more on that). For those workloads the GPU backend achieves dramatic speedups.

So, a two-layer transformer pass has no independent redexes at the GPU level. I knew that going in, but there was some part of me that was digging for gold to find some insights into maybe something I've missed from my assumption. At some point, I knew this project's potential was moreso a "let's learn Bend and it's pros and cons + learn more about LLMs at the structural level" and less about finding something novel. However, I had a lot of fun, some banging-against-the-wall moments and some new research ideas I want to use Bend in! Overall, good times were had! 
## running it

```bash
bend run-rs -s model.bend   # rust
bend run-c -s model.bend   # c
bend run-cu -s model.bend   # cuda
```

## notes
- Bend only has 24-bit floats (`f24`), so we can't be as precise yet. Most modern LLMs run on `f32` which gives 7 points of precision while`f24` only gives 3.
- If you dissect and run parts of this code. You'll see some output in raw lambda calculus form (eg. `λa (a FVec/Cons/tag 3.000 λb (b FVec/Cons/tag 4.000 FVec/Nil))`) , which is how HVM2 represents data structures internally before they get pretty-printed.
