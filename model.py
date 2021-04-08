from torch import nn


def model(num_inputs=14, num_initial_nodes=100):
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

        nn.Linear(50, 25),
        # nn.BatchNorm1d(25),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),

        nn.Linear(25, 10),
        # nn.BatchNorm1d(10),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),

        nn.Linear(10, 1),
        # nn.ReLU(),
    )
    return model
