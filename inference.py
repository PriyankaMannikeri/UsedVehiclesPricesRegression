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


# paramerters
num_initial_node = 600
num_features = 11
num_model_layers = 9
divide_by_max = True
standard_scaler = True
loss_mse = MSELoss()
loss = L1Loss()
batch_size = 32
lr = 1e-2
loss_mult = 1e-2

num_epochs = 400
log_loss_iter = 20000
evaluate_test_iter = 10000
save_model_iter = 10000
scheduler_iter = 4000

# data = np.load(open("./datasets/features_{}.npy".format(num_features), "rb"))
data = np.load(open("./datasets/filtered/features_{}.npy".format(num_features), "rb"))

if divide_by_max:
    data[:, 0:-1] = data[:, 0:-1] * 1. / np.max(data[:, 0:-1], axis=0)

if standard_scaler:
    trans = StandardScaler()
    data[:, 0:-1] = trans.fit_transform(data[:, 0:-1])

# train, test split
# shuffle data
kfold_test_value = 1
random_state = 2180
train_percentage = 0.9

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

# train_dataset = UsedVechiclesDataset(train_data, train_price)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)

test_dataset = UsedVechiclesDataset(test_data, test_price)
test_dataloader = DataLoader(test_dataset, batch_size=1)

regressor = model(num_inputs=num_features, num_initial_nodes=num_initial_node, num_model_layers=num_model_layers)
regressor = regressor.float()

# model_path = "./experiments/num-features-11-scalar-True-maxnorm-True-numnodes-800-NjiqOHIIwI/model-1190000.pth"
model_path = "./experiments/num-features-11-scalar-True-maxnorm-True-numnodes-600-trainSplit-0.9-datasetFiltered-True-num_model_layers-9-kfold_test_value-1-lE3ShykBkm/model-820000.pth"
regressor.load_state_dict(torch.load(model_path))

test_loss = 0.0
test_abs_loss = 0.0
test_mse_loss = 0.0
test_gt_values = []
test_pred_values = []
test_abs_loss_values = []

with torch.no_grad():
    regressor.eval()

    for idx1, test_sample in enumerate(test_dataloader):
        test_data = test_sample["data"].float()
        test_gt = test_sample["gt"].float()

        test_pred = regressor(test_data)
        # test_loss += loss(test_pred.flatten(), test_gt) * loss_mult
        test_loss += loss_mse(test_pred.flatten(), test_gt)
        test_curr_abs_loss = torch.abs(test_pred.flatten() - test_gt).float().item()
        test_abs_loss += test_curr_abs_loss

        test_curr_mse_loss = np.square((test_pred.flatten().numpy() - test_gt.numpy()))
        test_mse_loss += test_curr_mse_loss

        test_gt_values.append(test_gt.item())
        test_pred_values.append(test_pred.float().item())
        test_abs_loss_values.append(test_curr_abs_loss)

    test_loss = test_loss / len(test_dataloader.dataset)
    test_abs_loss = test_abs_loss / len(test_dataloader.dataset)
    test_mse_loss = np.sqrt(test_mse_loss / len(test_dataloader.dataset))
    print(f"TEST MSE LOSS: {test_loss}")
    print(f"TEST RMSE LOSS: {test_mse_loss}")
    print(f"TEST MAE LOSS: {test_abs_loss}")

# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots()
# ax.scatter(test_gt_values, test_pred_values, s=5, cmap=plt.cm.coolwarm)
# # ax.plot([0,1],[0,1], transform=ax.transAxes)
#
# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]
# # now plot both limits against eachother
# ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
# ax.set_aspect('equal')
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.set_title("Scatter plot of model prediction vs gt")
# # ax.set_axis_off()
# plt.xlabel("Ground truth ($)")
# plt.ylabel("Model Prediction ($)")
# plt.xlim([-5000, 55000])
# # plt.show()
# plt.savefig("./images/scatterplot_test_error_best_model.png", dpi=256, bbox_inches='tight')
# plt.close()
#
# plt.hist(test_abs_loss_values, bins=20)
# plt.ylabel('Count of examples')
# plt.xlabel('MAE Error')
# plt.title("Histogram of MAE error across test dataset")
# plt.show()
#
# print(np.max(test_abs_loss_values))
# print(np.min(test_abs_loss_values))
# print(np.mean(test_abs_loss_values))
# print(np.std(test_abs_loss_values))
# print(np.var(test_abs_loss_values))
