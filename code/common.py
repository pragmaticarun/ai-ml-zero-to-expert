"""Shared helpers for course scripts. Figures go to code/week-XX/out/."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def outdir(file):
    d = os.path.join(os.path.dirname(os.path.abspath(file)), "out")
    os.makedirs(d, exist_ok=True)
    return d

def savefig(file, name):
    p = os.path.join(outdir(file), name); plt.tight_layout(); plt.savefig(p, dpi=90); plt.close()
    print(f"[saved figure] {p}")

# ---- Month-2 dataset loaders (download on first use into data/cache) ----
import sys as _sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CACHE = os.path.join(_ROOT, "data", "cache")
def _dl(name):
    _sys.path.insert(0, os.path.join(_ROOT, "data")); import download; download.TARGETS[name]()

def load_california():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(data_home=CACHE, as_frame=True); return d.data, d.target

def load_adult():
    """Adult census income (OpenML 'adult' v2). Returns X (DataFrame, with NaNs), y (1 = income >50K)."""
    from sklearn.datasets import fetch_openml
    d = fetch_openml("adult", version=2, as_frame=True, data_home=CACHE)
    y = (d.target.astype(str).str.strip().str.startswith(">50K")).astype(int)
    return d.data, y

def load_bike():
    """Hourly bike-sharing demand (OpenML 42712), rows in time order. Returns the full DataFrame."""
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=42712, as_frame=True, data_home=CACHE, parser="liac-arff"); return d.frame

def load_movielens(split=1):
    import pandas as pd
    p = os.path.join(CACHE, "ml-100k")
    if not os.path.exists(p): _dl("movielens")
    cols = ["user", "item", "rating", "ts"]
    r = lambda f: pd.read_csv(os.path.join(p, f), sep="\t", names=cols)
    return r(f"u{split}.base"), r(f"u{split}.test")

def load_wholesale():
    import pandas as pd
    p = os.path.join(CACHE, "wholesale", "Wholesale customers data.csv")
    if not os.path.exists(p): _dl("wholesale")
    return pd.read_csv(p)

# ---- Month-3 loaders (images/text). All cache under data/cache; see data/README.md ----
def load_mnist(fashion=False):
    """MNIST or Fashion-MNIST via torchvision (downloads ~12/30 MB on first use).
    Returns uint8 arrays Xtr (60000,28,28), ytr, Xte (10000,28,28), yte."""
    import numpy as np
    from torchvision import datasets
    D = datasets.FashionMNIST if fashion else datasets.MNIST
    tr = D(CACHE, train=True, download=True); te = D(CACHE, train=False, download=True)
    return tr.data.numpy(), tr.targets.numpy(), te.data.numpy(), te.targets.numpy()

def load_cifar_small():
    """CIFAR-10 small (OpenML 40926: 20,000 images, 32x32x3). The cs.toronto.edu tarball was not reachable
    from our box, so we use this OpenML copy. Returns X uint8 (20000,32,32,3), y int64, class names."""
    import numpy as np
    p = os.path.join(CACHE, "cifar10_small.npz")
    if not os.path.exists(p):
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=40926, as_frame=False, data_home=CACHE, parser="liac-arff")
        X = d.data.astype(np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1); y = d.target.astype(np.int64)
        np.savez_compressed(p, X=X, y=y)
    z = np.load(p)
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return z["X"], z["y"], names

def load_shakespeare():
    """Tiny Shakespeare (~1.1 MB) from the char-rnn repository. Returns the text as str."""
    p = os.path.join(CACHE, "tinyshakespeare.txt")
    if not os.path.exists(p):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", p)
    return open(p, encoding="utf-8").read()
