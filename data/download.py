"""Dataset downloader/cacher. Usage: python data/download.py [california adult bike movielens wholesale | all]
Everything is cached in data/cache/. Loaders used by the scripts live in code/common.py."""
import sys, os, urllib.request, zipfile
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"); os.makedirs(CACHE, exist_ok=True)
def _zip(url, name, sub):
    if not os.path.exists(os.path.join(CACHE, sub)):
        z = os.path.join(CACHE, name); urllib.request.urlretrieve(url, z)
        zipfile.ZipFile(z).extractall(os.path.join(CACHE, sub) if sub == "wholesale" else CACHE)
    print(name, "ok in", CACHE)
def california():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(data_home=CACHE, as_frame=True); print("california:", d.frame.shape)
def adult():
    from sklearn.datasets import fetch_openml
    d = fetch_openml("adult", version=2, as_frame=True, data_home=CACHE); print("adult:", d.frame.shape)
def bike():
    from sklearn.datasets import fetch_openml   # OpenML id 42712; pandas parser fails on its bool column, so use liac-arff
    d = fetch_openml(data_id=42712, as_frame=True, data_home=CACHE, parser="liac-arff"); print("bike:", d.frame.shape)
def movielens(): _zip("https://files.grouplens.org/datasets/movielens/ml-100k.zip", "ml-100k.zip", "ml-100k")
def wholesale(): _zip("https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip", "wholesale.zip", "wholesale")
TARGETS = dict(california=california, adult=adult, bike=bike, movielens=movielens, wholesale=wholesale)
if __name__ == "__main__":
    args = sys.argv[1:] or ["california"]
    for t in (TARGETS if args == ["all"] else args): TARGETS[t]()
