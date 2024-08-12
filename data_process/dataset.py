import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

class MovieDS(Dataset):
    def __init__(self, rating_df):
        
        self.df_user = rating_df['userId']
        self.df_movie = rating_df['movieId']
        self.df_rating = rating_df['rating']

        assert (len(self.df_user) == len(self.df_user)), 'there is a big problem'
        
    
    def __getitem__(self, idx):
        return self.df_user.iloc[idx], self.df_movie.iloc[idx], self.df_rating.iloc[idx]

    def __len__(self):
        return len(self.df_user)



class MovieDataModule(object):
    def __init__(self, path_to_rating="ml-latest-small/ratings.csv", batch_size=8, split=0.5, device='mps'):
        self.path_to_rating = path_to_rating
        self.batch_size = batch_size
        
        self.rating_df = pd.read_csv(self.path_to_rating)

        self.unique_movie_set = set(self.rating_df['movieId'].unique())
        self.unique_user_set = set(self.rating_df['userId'].unique())

        # user_negative = self._negative_sample(self.rating_df)
        # self.rating_df = pd.concat([self.rating_df, user_negative], axis=0, ignore_index=True)


        self.rating_df = self._reindex(self.rating_df)

        if device == 'mps':
            print('LOG: device == mps converting to float32')
            self.rating_df['rating'] = self.rating_df['rating'].astype(np.float32)

        self.train_ds, self.test_ds = self._create_train_test(self.rating_df, train_proportion=split)


    def _create_train_test(self, df, train_proportion):
        return train_test_split(df, test_size=(1 - train_proportion), train_size=train_proportion)
    
    def _reindex(self, ratings):

        user_list = list(ratings['userId'].drop_duplicates())
        user2id = {w: i for i, w in enumerate(user_list)}

        item_list = list(ratings['movieId'].drop_duplicates())
        item2id = {w: i for i, w in enumerate(item_list)}

        ratings['userId'] = ratings['userId'].apply(lambda x: user2id[x])
        ratings['movieId'] = ratings['movieId'].apply(lambda x: item2id[x])
        return ratings

    def _negative_sample(self, df, sample_size=1):
        user_interact_with = (
            df.groupby('userId')['movieId']
            .apply(set)
            .reset_index()
            .rename(columns={'movieId': 'interactedItems'}))

        
        user_interact_with['negativeItems'] = user_interact_with['interactedItems'].apply(lambda x: self.unique_movie_set - x)
        user_interact_with['negativeSample'] = user_interact_with['negativeItems'].apply(lambda x: random.sample(list(x), sample_size)[0])

        reshape_to_ds = pd.DataFrame(zip(user_interact_with['userId'], user_interact_with['negativeSample'], [0] * len(user_interact_with['userId']),[0] * len(user_interact_with['userId'])), 
                                     columns=['userId', 'movieId', 'rating', 'timestamp'])

        return reshape_to_ds
    
    def write_df(self, df, filename_path='ml-latest-small/df.csv'):
        df.to_csv(filename_path)

    def train_dataloader(self):
        ds = MovieDS(self.train_ds)
        return DataLoader(ds, batch_size=self.batch_size)
    
    def test_dataloader(self):
        ds = MovieDS(self.test_ds)
        return DataLoader(ds, batch_size=self.batch_size)



if __name__ == '__main__':
    datamodule = MovieDataModule()

    train_ds = datamodule.train_dataloader()
    test_ds = datamodule.test_dataloader()

    for i, (user_id, item_id, rating) in enumerate(train_ds):
        print(f'log batch n{i}: user_id = {user_id}, item_id = {item_id}, rating = {rating}')
        if i == 5:
            break