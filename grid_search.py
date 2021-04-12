from joblib import Parallel, delayed, parallel_backend
from train import train


# def run_exp(num_feat):
def run_exp(exp):
    num_features, num_initial_node, filtered_dataset = exp
    print("running experiment with num_features: {}\nnum_initial_node: {}\nfiltered_dataset: {}\n".format(num_initial_node, num_initial_node, filtered_dataset))
    # train(11, num_initial_node)
    train(exp)

# num_features = [14, 12, 11, 10, 8, 5]
num_features = [14, 12, 11]
num_initial_nodes = [100, 200, 600, 800]
dataset_filtered = [True, False]
num_exp = 4

exps = []
for _ in range(num_exp):
    for n in num_features:
        for nodes in num_initial_nodes:
            for filtered in dataset_filtered:
                exps.append([n, nodes, filtered])

with parallel_backend("loky", inner_max_num_threads=1):
    # Parallel(n_jobs=6)(delayed(run_exp)(num_feat) for num_feat in exps)
    # Parallel(n_jobs=6)(delayed(run_exp)(num_initial_node) for num_initial_node in num_initial_nodes)
    Parallel(n_jobs=6)(delayed(run_exp)(exp) for exp in exps)
