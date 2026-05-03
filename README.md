# bend-gpt
A mock transformer language model written in [Bend](github.com/HigherOrderCO/Bend) and run on [HVM2](https://github.com/HigherOrderCO/HVM2).

## what is & why are we using bend

Bend is a programming language that looks like Python but runs in parallel on CPUs and GPUs automatically. You write your functions normally and it figures out what can run at the same time, which turns out to matter a lot for language models.

When a transformer processes a sequence of tokens, most of the work is independent. Each token does its own calculations, each attention head does its own thing, none of them need to wait for each other. Normally to take advantage of that you have to explicitly tell the hardware what to run in parallel, which is most of what CUDA code is doing. In Bend that happens automatically, so you get the speed benefits without the complexity. 

This project aims to experiment and understand what it looks like to build a transformer in Bend. The benefit is seeing how much Bend's automatic parallelization can accelerate model architecture without writing a single line of thread management code.

## running it

```bash
bend run-rs main.bend   # rust
bend run-c main.bend   # c
bend run-cu main.bend   # cuda
```

add `-s` after the `run-*` to any of those to print runtime stats (reductions, time, MIPS).

## notes
- Bend only has 24-bit floats (`f24`), so we can't be as precise yet. 
- If you dissect and run parts of this code. You'll see some output in raw lambda calculus form (eg. `λa (a FVec/Cons/tag 3.000 λb (b FVec/Cons/tag 4.000 FVec/Nil))`) , which is how HVM2 represents data structures internally before they get pretty-printed.

