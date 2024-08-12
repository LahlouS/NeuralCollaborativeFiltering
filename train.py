import sys
import torch
from data_process.dataset import MovieDataModule
from model.model import NeuMF
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils.utils as utils 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'LOG: running on {device}')

EPOCHS = 10
batch_size = 8

datamodule = MovieDataModule()
train_ds = datamodule.train_dataloader()
test_ds = datamodule.test_dataloader()

model = NeuMF(c_len=len(datamodule.unique_movie_set),
                u_len=len(datamodule.unique_user_set),
                embed_size=32, 
                layers=[64, 32, 16, 8, 4, 1])
model = model.to(device)

learning_rate = 0.0001 

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epchs in range(EPOCHS):
    print(f'STARTING EPOCH {epchs}')
    model.train()

    loss_track = []
    prog_bar = tqdm(train_ds)
    for idx, (user_idx, item_idx, rating) in enumerate(prog_bar):
        user_idx, item_idx, rating = user_idx.to(device), item_idx.to(device), rating.to(device)

        optimizer.zero_grad()
        pred_rating = model(item_idx, user_idx)
        loss = loss_function(pred_rating.squeeze(1), rating)
        loss.backward()
        optimizer.step()
        prog_bar.set_postfix(MSE=loss.item())
        loss_track.append(loss.item())
        if idx % 500 == 0:
            utils.plot_loss(loss_track, 'log/', f'loss{epchs}')

    
    model.eval()
    ###########################
        # TODO 
        # MAKE ALL THE EVAL LOGIC WITH METRICS AND STUFF

