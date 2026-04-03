## 2024-05-25 - PyTorch Vectorization Speedup
**Learning:** In PyTorch models, using a `for` loop over tensor elements is a severe performance bottleneck. Vectorizing operations like the Q-value target computation using `torch.where` and advanced indexing can yield a massive speedup (e.g., 60x faster for a batch size of 1000).
**Action:** Always look for `for` loops in hot paths like the training loop or prediction loop in PyTorch code. Replace them with vectorized operations (e.g. `torch.where`, `torch.max(..., dim=...)`, or advanced tensor indexing) to improve performance significantly.
