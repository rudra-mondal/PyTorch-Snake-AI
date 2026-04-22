## 2024-05-25 - PyTorch Vectorization Speedup
**Learning:** In PyTorch models, using a `for` loop over tensor elements is a severe performance bottleneck. Vectorizing operations like the Q-value target computation using `torch.where` and advanced indexing can yield a massive speedup (e.g., 60x faster for a batch size of 1000).
**Action:** Always look for `for` loops in hot paths like the training loop or prediction loop in PyTorch code. Replace them with vectorized operations (e.g. `torch.where`, `torch.max(..., dim=...)`, or advanced tensor indexing) to improve performance significantly.

## 2024-05-28 - PyTorch Tensor Creation Performance from Tuples
**Learning:** In PyTorch, creating a tensor directly from a tuple or list of NumPy arrays (e.g., `torch.tensor(tuple_of_arrays)`) is extremely slow and triggers an internal PyTorch warning.
**Action:** When converting batches of data stored as lists/tuples of NumPy arrays into PyTorch tensors, always convert the list/tuple to a single NumPy array first (`torch.tensor(np.array(tuple_of_arrays))`). In this project, it sped up batch processing in `train_long_memory` by approximately 8x.
