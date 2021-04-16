from joblib import Parallel, delayed, parallel_backend
from train import train


# def run_exp(num_feat):
def run_exp(exp):
    num_features, num_initial_node, filtered_dataset, num_model_layers, kfold_test_value = exp
    print("running experiment with num_features: {}\nnum_initial_node: {}\nfiltered_dataset: {}\nnum_model_layers: {}\nkfold_test_value: {}\n".format(num_initial_node, num_initial_node, filtered_dataset, num_model_layers,kfold_test_value))
    # train(11, num_initial_node)
    train(exp)

# num_features = [14, 12, 11, 10, 8, 5]
num_features = [11]
num_initial_nodes = [600]
num_model_layers = [9]
dataset_filtered = [True]
num_exp = 4
kfold_test_values = list(range(1, 11))

exps = []
for _ in range(num_exp):
    for n in num_features:
        for nodes in num_initial_nodes:
            for filtered in dataset_filtered:
                for layers in num_model_layers:
                    for kfold_test_value in kfold_test_values:
                        exps.append([n, nodes, filtered, layers, kfold_test_value])

with parallel_backend("loky", inner_max_num_threads=1):
    # Parallel(n_jobs=6)(delayed(run_exp)(num_feat) for num_feat in exps)
    # Parallel(n_jobs=6)(delayed(run_exp)(num_initial_node) for num_initial_node in num_initial_nodes)
    Parallel(n_jobs=6)(delayed(run_exp)(exp) for exp in exps)
