import torch
import numpy as np
from model import model
from sklearn.preprocessing import StandardScaler
from dataloader import UsedVechiclesDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import MSELoss, L1Loss
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import subprocess
import shlex
import random
import string
import json
import os


def launch_tb():
    cmd = "tensorboard --logdir ./experiments --port 12345"
    subprocess.Popen(shlex.split(cmd),
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, shell=True)
    # _ = p.communicate()


def train(job):
    # paramerters
    num_features, num_initial_node, filtered_dataset, num_model_layers, kfold_test_value = job
    divide_by_max = True
    standard_scaler = True
    # loss = MSELoss()
    loss = L1Loss()
    batch_size = 32
    lr = 1e-2
    loss_mult = 1e-2

    num_epochs = 300
    log_loss_iter = 20000
    evaluate_test_iter = 20000
    save_model_iter = 20000
    scheduler_iter = 4000
    train_percentage = 0.9

    if kfold_test_value is None:
        random_state = random.randint(1, 99999)
    else:
        random_state = 2180

    launch_tb()

    # generate random experiment tracking id
    source = string.ascii_letters + string.digits
    expt_id = ''.join((random.choice(source) for i in range(10)))
    expt_name = "num-features-{}-scalar-{}-maxnorm-{}-numnodes-{}-trainSplit-{}-datasetFiltered-{}-num_model_layers-{}-onehot-True-kfold_test_value-{}-{}".\
        format(num_features, standard_scaler, divide_by_max, num_initial_node, train_percentage, filtered_dataset, num_model_layers, kfold_test_value, expt_id)
    writer = SummaryWriter("./experiments/{}".format(expt_name))

    # dumpy json file
    parameters = {
        "num_features": num_features, "num_initial_node": num_initial_node, "filtered_dataset": filtered_dataset,
        "divide_by_max": divide_by_max, "standard_scaler": standard_scaler, "batch_size": batch_size,
        "lr": lr, "loss_mult": loss_mult, "num_epochs": num_epochs, "evaluate_test_iter": evaluate_test_iter,
        "train_percentage": train_percentage, "random_state": random_state, "num_model_layers": num_model_layers, "kfold_test_value": kfold_test_value,
    }
    with open('./experiments/{}/parameters.json'.format(expt_name), 'w', encoding='utf-8') as f:
        json.dump(parameters, f, ensure_ascii=False, indent=4)

    if not filtered_dataset:
        data = np.load(open("./datasets/features_{}.npy".format(num_features), "rb"), allow_pickle=True)
    elif filtered_dataset:
        # data = np.load(open("./datasets/filtered/features_{}.npy".format(num_features), "rb"), allow_pickle=True)
        data = np.load(open("./datasets/ordered-label-encoding/features_{}.npy".format(num_features), "rb"), allow_pickle=True)
    # data = np.load(open("./datasets/onehot-encoded/features_{}.npy".format(num_features), "rb"), allow_pickle=True)

    if divide_by_max:
        data[:, 0:-1] = data[:, 0:-1] * 1. / np.max(data[:, 0:-1], axis=0)

    if standard_scaler:
        trans = StandardScaler()
        data[:, 0:-1] = trans.fit_transform(data[:, 0:-1])

    # train, test split
    # shuffle data
    data = shuffle(data, random_state=random_state)
    train_idx = int(train_percentage*data.shape[0])
    if kfold_test_value is None:
        train_data, train_price = data[0:train_idx, 0:-1], data[0:train_idx, -1]
        test_data, test_price = data[train_idx:, 0:-1], data[train_idx:, -1]
    else:
        test_length = data.shape[0] - train_idx
        train_data = np.concatenate(
            (
                data[0:(kfold_test_value-1)*test_length, 0:-1],
                data[kfold_test_value*test_length:, 0:-1]
            )
        )
        train_price = np.concatenate(
            (
                data[0:(kfold_test_value-1)*test_length, -1],
                data[kfold_test_value*test_length:, -1]
            )
        )
        test_data = data[(kfold_test_value-1)*test_length:kfold_test_value*test_length, 0:-1]
        test_price = data[(kfold_test_value-1)*test_length:kfold_test_value*test_length, -1]

    train_dataset = UsedVechiclesDataset(train_data, train_price)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)

    test_dataset = UsedVechiclesDataset(test_data, test_price)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    regressor = model(num_inputs=num_features, num_initial_nodes=num_initial_node, num_model_layers=num_model_layers)
    regressor = regressor.float()

    optimizer = optim.Adam(regressor.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    iter = 0
    best_test_mae = float("inf")

    regressor.train()
    print(f"len of train: {len(train_dataloader.dataset)} | len of test: {len(test_dataloader.dataset)}")
    for epoch in range(num_epochs):
        print(f"epoch:{epoch}")

        for idx, batch in enumerate(train_dataloader):
            regressor.train()
            optimizer.zero_grad()

            data = batch["data"].float()
            gt = batch["gt"].float()

            pred = regressor(data).float()
            curr_loss = loss(pred.flatten(), gt) * loss_mult

            curr_loss.backward()
            optimizer.step()

            if iter % log_loss_iter == 0:
                print(f"iter:{iter} | train_loss: {curr_loss}")
                print(f"gt:{gt[0]} | pred: {pred[0].float().item()}")
                print(f"gt:{gt[1]} | pred: {pred[1].float().item()}")
                writer.add_scalar("Train/Loss", curr_loss, iter)
                writer.flush()

            if iter % evaluate_test_iter == 0 and iter > 0:
                print("starting test..")
                test_loss = 0.0
                test_abs_loss = 0.0
                test_gt_values = []
                test_pred_values = []
                test_abs_loss_values = []

                with torch.no_grad():
                    regressor.eval()

                    for idx1, test_sample in enumerate(test_dataloader):
                        test_data = test_sample["data"].float()
                        test_gt = test_sample["gt"].float()

                        test_pred = regressor(test_data)
                        test_loss += loss(test_pred.flatten(), test_gt) * loss_mult
                        test_curr_abs_loss = torch.abs(test_pred.flatten() - test_gt).float().item()
                        test_abs_loss += test_curr_abs_loss
                        if test_curr_abs_loss < best_test_mae:
                            # Write-Overwrites
                            file1 = open("./experiments/" + expt_name + "/best_test_mae.txt", "a")#write mode
                            file1.write("{} @{}".format(test_curr_abs_loss, iter))
                            file1.close()

                        test_gt_values.append(test_gt.item())
                        test_pred_values.append(test_pred.float().item())
                        test_abs_loss_values.append(test_curr_abs_loss)

                    test_loss = test_loss / len(test_dataloader.dataset)
                    test_abs_loss = test_abs_loss / len(test_dataloader.dataset)
                    print(f"TEST MSE LOSS: {test_loss}")
                    print(f"TEST ABS LOSS: {test_abs_loss}")
                    writer.add_scalar("TEST/MAE", test_abs_loss, iter)
                    top5_loss_idx = np.argsort(test_abs_loss_values)[-5:]
                    for test_idx in top5_loss_idx:
                        print(test_idx, test_abs_loss_values[test_idx], test_gt_values[test_idx], test_pred_values[test_idx])

            iter += 1

            if iter % scheduler_iter == 0 and iter > 0:
                scheduler.step()
                writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], iter)

            if iter % save_model_iter == 0 and iter > 0:
                # TODO: create a random experiment run and save hyper-parameters in a json file
                torch.save(regressor.state_dict(), "./experiments/{}/model-{}.pth".format(expt_name, iter))

            # # ONNX for netron Viz
            # regressor.train()
            # input_names = ['features']
            # output_names = ['price']
            # torch.onnx.export(regressor, batch["data"].float(), './models/model.onnx', input_names=input_names, output_names=output_names)
