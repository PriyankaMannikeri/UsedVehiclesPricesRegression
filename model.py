from torch import nn


def model():
    # define model architecture
    model = nn.Sequential(
        nn.Linear(14, 100),
        # nn.BatchNormd(500),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),

        nn.Linear(100, 100),
        # nn.BatchNorm1d(200),
        nn.LeakyReLU(),
        # nn.Dropout(p=0.1),

        nn.Linear(100, 50),
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
