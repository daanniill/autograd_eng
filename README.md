# micrograd

A from-scratch autograd engine and neural network library, inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). Built to understand backpropagation by implementing it at the level of individual scalar operations.

## What's implemented

- **`engine.py`** — `Value`: a scalar wrapper that tracks a computation graph and supports `+`, `-`, `*`, `/`, `**`, `tanh`, and `exp`, each with its own local backward pass. Calling `.backward()` topologically sorts the graph and runs reverse-mode autodiff to populate `.grad` on every node.
- **`nn.py`** — `Neuron`, `Layer`, and `MLP` classes built entirely on top of `Value`, with a `.parameters()` method on each for gradient-based optimization.
- **`demo.ipynb`** — trains the MLP on a `make_moons` binary classification dataset, with:
  - mean squared error loss
  - steepest descent
  - backtracking line search using the **Armijo condition** to pick a step size each iteration
  - a plot of the learned decision boundary

## Project structure

```
engine.py    # Value class + autograd
nn.py        # Neuron / Layer / MLP
demo.ipynb   # training demo on make_moons
notes/       # handwritten/derivation notes
```

## Notes

📄 [**notes/Micrograd.pdf**](notes/Micrograd.pdf) — derivations and notes worked through while building this (chain rule, gradient derivations for each op, line search). Click to open — GitHub renders PDFs in-browser, no download needed.
