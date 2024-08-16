import os
import sys
import torch
from data_process.dataset import MovieDataModule
from model.model import NeuMF
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils.utils as utils
import pandas as pd

device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
print(f'LOG: running on {device}')

EPOCHS = 200
batch_size = 512

datamodule = MovieDataModule(batch_size=batch_size)
train_ds = datamodule.train_dataloader()
test_ds = datamodule.test_dataloader()

model = NeuMF(c_len=len(datamodule.unique_movie_set),
                u_len=len(datamodule.unique_user_set),
                embed_size=32, 
                layers=[64, 32, 16, 8, 4, 1],
                mlp_weights='checkpoints/mlp_weights.ckpt',
                gmf_weights='checkpoints/mf_weights.ckpt')

model = model.to(device)

learning_rate = 0.001 

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_track = []
loss_track_test = []
for epchs in range(EPOCHS):
    print(f'STARTING EPOCH {epchs}')
    model.train()

    prog_bar = tqdm(train_ds)
    for idx, (user_idx, item_idx, rating) in enumerate(prog_bar):
        user_idx, item_idx, rating = user_idx.to(device), item_idx.to(device), rating.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        pred_rating = model(item_idx, user_idx)
        loss = loss_function(pred_rating.squeeze(1), rating)
        loss.backward()
        optimizer.step()
        prog_bar.set_postfix(MSE=loss.item())
        loss_track.append(loss.item())
    print(f'min loss epoch {epchs}:{min(loss_track)}')
    if epchs % 50 == 0:
        utils.plot_loss({ 'NeuMF_train': loss_track }, 'log/', f'loss_e{epchs}')

    
    model.eval()
    prog_bar = tqdm(test_ds)
    output_test_df = pd.DataFrame(columns=['userId', 'movieId', 'label', 'preds'])
    for idx, (user_idx, item_idx, rating) in enumerate(prog_bar):

        user_idx, item_idx, rating = user_idx.to(device), item_idx.to(device), rating.type(torch.FloatTensor).to(device)
        pred_rating = model(item_idx, user_idx)
        pred_rating = pred_rating.squeeze(1)
        loss_test = loss_function(pred_rating, rating)
        prog_bar.set_postfix(MSE=loss_test.item())
        loss_track_test.append(loss_test.item())
        if epchs % 50 == 0:
            df = pd.DataFrame({
                'userId': user_idx.detach().cpu(), 
                'movieId': item_idx.detach().cpu(), 
                'label': rating.detach().cpu(),
                'preds': pred_rating.detach().cpu()
            })
            output_test_df = pd.concat([output_test_df, df])

    if epchs % 50 == 0:
        utils.plot_loss({ 'NeuMF_test': loss_track_test }, 'log/', f'loss_test_e{epchs}')
        utils.to_csv(output_test_df, './preds/', f'preds_e_{epchs}.csv')

path = './checkpoints/'
utils.save_model_state(model, path, 'neumf_weights_1.ckpt')
