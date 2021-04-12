from torch import nn


def model(num_inputs=14, num_initial_nodes=100, num_model_layers=None):
    # define model architecture
    model = nn.Sequential(
        nn.Linear(num_inputs, max(100, num_initial_nodes)),
        # nn.BatchNormd(500),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),

        nn.Linear(max(100, num_initial_nodes), max(100, num_initial_nodes//2)),
        # nn.BatchNorm1d(200),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),

        nn.Linear(max(100, num_initial_nodes//2), 50),
        # nn.BatchNorm1d(100),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),

        nn.Linear(50, 50),
        # nn.BatchNorm1d(50),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),
    )

    if num_model_layers == 9 or num_model_layers == 11:
        num_50_layers = (num_model_layers - 7) // 2
        for idx in range(num_50_layers):
            model.add_module("50_50layer_{}".format(idx), nn.Sequential(
                nn.Linear(50, 50),
                nn.LeakyReLU(),
            ))

    model.add_module("middle_layer", nn.Sequential(
        nn.Linear(50, 25),
        # nn.BatchNorm1d(25),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),
    ))

    if num_model_layers == 9 or num_model_layers == 11:
        num_25_layers = (num_model_layers - 7) // 2
        for idx in range(num_25_layers):
            model.add_module("25_25layer_{}".format(idx), nn.Sequential(
                nn.Linear(25, 25),
                nn.LeakyReLU(),
            ))

    model.add_module("final_layer", nn.Sequential(
        nn.Linear(25, 10),
        # nn.BatchNorm1d(10),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),

        nn.Linear(10, 1),
        # nn.ReLU(),
    ))
    return model
