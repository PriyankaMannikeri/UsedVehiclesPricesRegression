from joblib import Parallel, delayed, parallel_backend
from train import train


# def run_exp(num_feat):
def run_exp(num_initial_node):
    print("running experiment with num_features: {}".format(num_initial_node))
    train(11, num_initial_node)

num_features = [14, 12, 11, 10, 8, 5]
num_initial_nodes = [100, 200, 400, 600, 800, 1000]
num_exp = 1

exps = []
for _ in range(num_exp):

    for n in num_features:
        exps.append(n)

with parallel_backend("loky", inner_max_num_threads=1):
    Parallel(n_jobs=6)(delayed(run_exp)(num_feat) for num_feat in exps)
    Parallel(n_jobs=6)(delayed(run_exp)(num_initial_node) for num_initial_node in num_initial_nodes)
