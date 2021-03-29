import torch
import numpy as np
from model import model
from sklearn.model_selection import train_test_split
from dataloader import UsedVechiclesDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import MSELoss, L1Loss
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import subprocess
import shlex


def launch_tb():
    cmd = "tensorboard --logdir ./tb-logs --port 12345"
    subprocess.Popen(shlex.split(cmd),
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, shell=True)
    # _ = p.communicate()

launch_tb()
writer = SummaryWriter("./tb-logs/")

data = np.load(open("features.npy", "rb"))
# todo see this norm
# data[:, 0:-1] = data[:, 0:-1] / np.linalg.norm(data[:, 0:-1])
data[:, 0:-1] = data[:, 0:-1] * 1. / np.max(data[:, 0:-1], axis=0)

# train, test split
# shuffle data
data = shuffle(data)
# np.random.shuffle(data)

train_idx = int(0.9*data.shape[0])

train_dataset = UsedVechiclesDataset(data[0:train_idx, 0:-1], data[0:train_idx, -1])
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True, drop_last=True)

test_dataset = UsedVechiclesDataset(data[train_idx:, 0:-1], data[train_idx:, -1])
test_dataloader = DataLoader(test_dataset, batch_size=1)

regressor = model()
regressor = regressor.float()

optimizer = optim.Adam(regressor.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
# loss = MSELoss()
loss = L1Loss()
loss_mult = 1e-2

num_epochs = 1000
log_loss_iter = 20000
evaluate_test_iter = 20000
save_model_iter = 20000
scheduler_iter = 4000
iter = 0

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
            writer.add_scalar("Train/MSE", curr_loss, iter)
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
            torch.save(regressor.state_dict(), "./models/model-{}.pth".format(iter))

        # # ONNX for netron Viz
        # regressor.train()
        # input_names = ['features']
        # output_names = ['price']
        # torch.onnx.export(regressor, batch["data"].float(), './models/model.onnx', input_names=input_names, output_names=output_names)
