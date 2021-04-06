from joblib import Parallel, delayed, parallel_backend
from train import train

def run_exp(num_feat):
    print("running experiment with num_features: {}".format(num_feat))
    train(num_feat)

num_features = [14, 12, 11, 10, 8, 5]

num_exp = 2

exps = []
for _ in range(num_exp):

    for n in num_features:
        exps.append(n)

with parallel_backend("loky", inner_max_num_threads=1):
    Parallel(n_jobs=6)(delayed(run_exp)(num_feat) for num_feat in exps)
