import sys
import os
import torch
from data_process.dataset import MovieDataModule
from model.model import MultiLayerPerceptron, GeneralisedMatrixFactorization
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils.utils as utils 

print('*** LOG: pre-training MF and MLP ***')
condition = torch.cuda.is_available()
print('LOG -------->', condition)
device = torch.device("cuda" if condition else "cpu")
print(f'LOG: running on {device}')

EPOCHS = 150
batch_size = 512

datamodule = MovieDataModule(batch_size=batch_size)
train_ds = datamodule.train_dataloader()
test_ds = datamodule.test_dataloader()


model_mf = GeneralisedMatrixFactorization(c_len=len(datamodule.unique_movie_set), u_len=len(datamodule.unique_user_set), embed_size=32)
model_mlp = MultiLayerPerceptron(c_len=len(datamodule.unique_movie_set), u_len=len(datamodule.unique_user_set), embed_size=32)

model_mf = model_mf.to(device)
model_mlp = model_mlp.to(device)

learning_rate = 0.001 

loss_function = torch.nn.MSELoss()

optimizer_mf = torch.optim.Adam(model_mf.parameters(), lr=learning_rate)
optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=learning_rate)

for epchs in range(EPOCHS):
    print(f'STARTING EPOCH {epchs}')
    model_mf.train()
    model_mlp.train()

    loss_track_mf = []
    loss_track_mlp = []

    prog_bar = tqdm(train_ds)
    for idx, (user_idx, item_idx, rating) in enumerate(prog_bar):
        user_idx, item_idx, rating = user_idx.to(device), item_idx.to(device), rating.type(torch.FloatTensor).to(device)

        optimizer_mf.zero_grad()
        optimizer_mlp.zero_grad()
        
        pred_rating_mf = model_mf(item_idx, user_idx)
        pred_rating_mlp = model_mlp(item_idx, user_idx)
        
        loss_mf = loss_function(pred_rating_mf.squeeze(1), rating)
        loss_mlp = loss_function(pred_rating_mlp.squeeze(1), rating)

        loss_mf.backward()
        loss_mlp.backward()

        optimizer_mf.step()
        optimizer_mlp.step()

        prog_bar.set_postfix(MSE_mf=loss_mf.item(), MSE_mlp=loss_mlp.item())

        loss_track_mf.append(loss_mf.item())
        loss_track_mlp.append(loss_mlp.item())

    print(f'min loss epoch_{epchs}: MF_{min(loss_track_mf)} | MLP_{min(loss_track_mlp)}')
    utils.plot_loss({'MF_loss': loss_track_mf, 'MLP_loss': loss_track_mlp}, 'log/', f'pre_train_loss_e{epchs}')

path = './checkpoints/'
utils.save_model_state(model_mf, path, 'mf_weights.ckpt')
utils.save_model_state(model_mlp, path, 'mlp_weights.ckpt')


