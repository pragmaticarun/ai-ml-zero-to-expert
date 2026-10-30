"""Day 83 (optional): JAX tour on CPU: pure functions, grad, jit, vmap, explicit PRNG keys."""
import time, numpy as np, jax, jax.numpy as jnp
print("jax", jax.__version__, "devices", jax.devices())
f = lambda x, y: x**2 * y + 3 * y; print("df/dx, df/dy at (2,1):", [float(v) for v in jax.grad(f, argnums=(0, 1))(2.0, 1.0)])
key = jax.random.PRNGKey(0); k1, k2, k3 = jax.random.split(key, 3)
X = jax.random.normal(k1, (2000, 5)); w_true = jnp.arange(1.0, 6.0); y = X @ w_true + 0.1 * jax.random.normal(k2, (2000,))
loss = lambda w: jnp.mean((X @ w - y) ** 2); g = jax.grad(loss); w = jnp.zeros(5)
gj = jax.jit(g); step = jax.jit(lambda w: w - 0.1 * g(w))
for _ in range(200): w = step(w)
print("fitted w:", np.round(np.asarray(w), 3), "loss", float(loss(w)))
print("grad check vs analytic 2X^T(Xw-y)/n, max diff:", float(jnp.abs(g(w) - 2 * X.T @ (X @ w - y) / len(y)).max()))
def mlp(params, x):
    for W, b in params[:-1]: x = jnp.tanh(x @ W + b)
    W, b = params[-1]; return x @ W + b
ks = jax.random.split(k3, 4); sizes = [5, 256, 256, 256, 1]; params = [(jax.random.normal(k, (a, b)) / np.sqrt(a), jnp.zeros(b)) for k, a, b in zip(ks, sizes[:-1], sizes[1:])]
L = lambda p: jnp.mean((mlp(p, X)[:, 0] - y) ** 2); gL = jax.grad(L); gLj = jax.jit(gL)
jax.block_until_ready(gLj(params))
def bench(fn, n=20):
    jax.block_until_ready(fn(params)); t = time.perf_counter()
    for _ in range(n): jax.block_until_ready(fn(params))
    return (time.perf_counter() - t) / n * 1e3
print(f"MLP grad: eager {bench(gL):.2f} ms, jit {bench(gLj):.2f} ms per call")
fresh = jax.jit(lambda p: jax.grad(L)(p)); t = time.perf_counter(); jax.block_until_ready(fresh(params)); print(f"first call of a fresh jit (trace + compile + run): {(time.perf_counter()-t)*1e3:.0f} ms")
per_ex = jax.vmap(jax.grad(lambda w, x, t: (x @ w - t) ** 2), in_axes=(None, 0, 0))(w, X[:4], y[:4]); print("per-example grads via vmap, shape", per_ex.shape)
print("mean of per-example grads equals batch grad:", bool(jnp.allclose(per_ex.mean(0), jax.grad(lambda w: jnp.mean((X[:4] @ w - y[:4]) ** 2))(w), atol=1e-5)))
c = 0
def impure(x): global c; c += 1; return x * 2
ji = jax.jit(impure); ji(1.0); ji(2.0); ji(3.0); print("python side effect ran", c, "time(s) for 3 jitted calls (tracing only)")
