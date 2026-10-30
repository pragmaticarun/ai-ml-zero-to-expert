"""Day 2: NumPy arrays, dtypes, shapes, indexing, broadcasting."""
import numpy as np
rng = np.random.default_rng(0)
a = np.array([1, 2, 3], dtype=np.float64)
M = np.arange(12).reshape(3, 4)
print("a:", a, a.dtype, a.shape); print("M:\n", M, M.shape, "strides(bytes):", M.strides)
print("M[1]:", M[1], " M[:,2]:", M[:, 2], " M[M>6]:", M[M > 6])
print("view vs copy: M.T shares memory?", np.shares_memory(M, M.T), "| M.copy()?", np.shares_memory(M, M.copy()))
# Broadcasting: (3,4) + (4,) -> row-wise; (3,4) + (3,1) -> column-wise
print("M + [10,20,30,40]:\n", M + np.array([10, 20, 30, 40]))
print("M - row means (keepdims):\n", M - M.mean(axis=1, keepdims=True))
# Standardise a feature matrix in one line
X = rng.normal(loc=[50, 3], scale=[10, 1], size=(1000, 2))
Z = (X - X.mean(axis=0)) / X.std(axis=0)
print("before mean/std:", X.mean(0).round(2), X.std(0).round(2), " after:", Z.mean(0).round(6), Z.std(0).round(6))
# Common bug: shape (n,) vs (n,1)
y = np.arange(3); col = y.reshape(-1, 1)
print("y - col shape (accidental broadcast!):", (y - col).shape)
# dtype pitfalls
i8 = np.array([100, 120], dtype=np.int8); print("int8 overflow 100+120 ->", (i8[0] + i8[1]))
print("float32 eps", np.finfo(np.float32).eps, "float64 eps", np.finfo(np.float64).eps)
