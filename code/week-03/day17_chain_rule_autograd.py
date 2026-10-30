"""Day 17: chain rule and a ~40-line scalar reverse-mode autograd (micrograd-style, written from scratch)."""
import math
class Value:
    def __init__(self, data, parents=(), op=""):
        self.data, self.grad, self._parents, self._back, self.op = data, 0.0, parents, lambda: None, op
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o); out = Value(self.data + o.data, (self, o), "+")
        def back(): self.grad += out.grad; o.grad += out.grad
        out._back = back; return out
    __radd__ = __add__
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o); out = Value(self.data * o.data, (self, o), "*")
        def back(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._back = back; return out
    __rmul__ = __mul__
    def __sub__(self, o): return self + (-1) * o
    def __pow__(self, k):
        out = Value(self.data**k, (self,), f"**{k}")
        def back(): self.grad += k * self.data**(k-1) * out.grad
        out._back = back; return out
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def back(): self.grad += out.data * out.grad
        out._back = back; return out
    def backward(self):
        order, seen = [], set()
        def topo(v):
            if v not in seen:
                seen.add(v); [topo(p) for p in v._parents]; order.append(v)
        topo(self); self.grad = 1.0
        for v in reversed(order): v._back()
# f(x, y) = (x*y + exp(x))^2 ; check against hand-derived chain rule
x, y = Value(0.5), Value(-1.2); u = x*y + x.exp(); f = u**2; f.backward()
uu = 0.5*-1.2 + math.exp(0.5)
print(f"f={f.data:.5f}  df/dx autograd={x.grad:.5f} hand={2*uu*(-1.2 + math.exp(0.5)):.5f}  df/dy autograd={y.grad:.5f} hand={2*uu*0.5:.5f}")
# Train w, b on the Day-1 houses with this autograd
xs, ys = [6, 8, 10, 12, 15, 18, 20], [32, 41, 50, 58, 74, 88, 97]
w, b = Value(0.0), Value(0.0)
for step in range(2001):
    loss = sum(((w*xi + b) - yi)**2 for xi, yi in zip(xs, ys)) * (1/len(xs))
    w.grad = b.grad = 0.0; loss.backward()
    w.data -= 0.004 * w.grad; b.data -= 0.004 * b.grad * 25   # larger step for b: a hint at feature scaling (Day 25)
    if step % 500 == 0: print(f"step {step}: loss={loss.data:.3f} w={w.data:.3f} b={b.data:.3f}")
