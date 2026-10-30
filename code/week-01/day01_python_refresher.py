"""Day 1: Python for ML refresher + your first 'model' (a hand-tuned line)."""
from dataclasses import dataclass

# Tiny dataset: house size (100s of sq ft) -> price (lakh INR). Invented for teaching.
sizes  = [6, 8, 10, 12, 15, 18, 20]
prices = [32, 41, 50, 58, 74, 88, 97]

@dataclass
class Line:
    w: float
    b: float
    def predict(self, x): return self.w * x + self.b

def mse(model, xs, ys):
    return sum((model.predict(x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

# Comprehensions, zip, enumerate, f-strings: idioms used all course long
pairs = [(x, y) for x, y in zip(sizes, prices) if x >= 10]
print("pairs with size>=10:", pairs)
for i, (x, y) in enumerate(zip(sizes, prices)):
    print(f"house {i}: size={x:>3} price={y:>3}  price/size={y/x:.2f}")

# Try a few guesses by hand: this is 'learning' done manually
for w, b in [(4, 10), (5, 0), (4.8, 3), (4.6, 4.5)]:
    m = Line(w, b); print(f"w={w:<4} b={b:<4} MSE={mse(m, sizes, prices):8.3f}")

# Brute-force grid search: dumb but honest optimisation
best = min(((mse(Line(w/10, b/10), sizes, prices), w/10, b/10)
            for w in range(30, 70) for b in range(-50, 100)), key=lambda t: t[0])
print(f"grid search best: MSE={best[0]:.3f} w={best[1]} b={best[2]}")
print("predict 14 (1400 sq ft):", round(Line(best[1], best[2]).predict(14), 2), "lakh")
