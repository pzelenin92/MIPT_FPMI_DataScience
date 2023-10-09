import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm.auto import tqdm

# os CUDA_LAUNCH_BLOCKING=1, необходимо для отладки, чтобы увидеть, где происходит ошибка
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

data = Dataset.load_builtin("ml-100k", prompt=False)
df = pd.DataFrame(data.raw_ratings, columns=["userId", "movieId", "rating", "timestamp"])
df = df.sort_values("timestamp")
# conver timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

# convert userId and movieId to int
df["userId"] = df["userId"].astype(int)
df["movieId"] = df["movieId"].astype(int)

num_unique_users, num_unique_movies = df['userId'].nunique(), df['movieId'].nunique()
print(f"Number of unique users: {num_unique_users}")
print(f"Number of unique movies: {num_unique_movies}")

print('Dataframe:\n', df.head())

# split dataframe into train and test in 80:20 ratio
train_df = df.iloc[:int(len(df)*0.8)]
test_df = df.iloc[int(len(df)*0.8):]


# get df userId qnique values as pd.Series
unique_users = df['userId'].unique()
# get df  qnique moviId as pd.Series
unique_movies = df['movieId'].unique()

# print('Unique users:\n', unique_users)
print('Unique users shape:', unique_users.shape)
# print('Unique movies:\n', unique_movies)
print('Unique movies shape:', unique_movies.shape)

'''
Crate dataloader for train and test
'''
class movielens_dataset(Dataset):
    def __init__(self, df, qnique_users, qnique_movies):
        self.df = df
        self.unique_users = qnique_users
        self.unique_movies = qnique_movies

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # получаем user_id этого взаимодействия юзера с фильмом
        user_id = self.df.loc[idx, 'userId']
        # получаем movie_id этого взаимодействия юзера с фильмом
        movie_id = self.df.loc[idx, 'movieId']
        # находим индекс user_id в массиве уникальных user_id
        user_emb_idx = np.where(self.unique_users == user_id)[0][0]
        # находим индекс movie_id в массиве уникальных movie_id
        movie_emb_idx = np.where(self.unique_movies == movie_id)[0][0]
        # конвертируем их в torch.LongTensor для нахождения нужного эмбеддинга
        user_emb_idx = torch.LongTensor([user_emb_idx])
        movie_emb_idx = torch.LongTensor([movie_emb_idx])
        # получаем рейтинг этого взаимодействия юзера с фильмом
        ratings = torch.tensor(self.df.loc[idx, 'rating'].astype(np.float32))
        # возвращаем индексы эмбеддингов и рейтинг
        return user_emb_idx, movie_emb_idx, ratings

# create dataloaders for train and test
train_dataset = movielens_dataset(train_df.reset_index(drop=True), qnique_users=unique_users, qnique_movies=unique_movies)
test_dataset = movielens_dataset(test_df.reset_index(drop=True), qnique_users=unique_users, qnique_movies=unique_movies)

print(f'Number of train samples: {len(train_dataset)}\nNumber of test samples: {len(test_dataset)}')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

'''
Create model which takes
user_emb_idx and movie_emb_idx as input as torch long tensors
passes them through embedding layers,
concatenates the output of embedding layers
and passes it through 2 linear layer with relu activation
'''
class movielens_model(nn.Module):
    def __init__(self, num_unique_users, num_unique_movies):
        super(movielens_model, self).__init__()
        self.user_embed = nn.Embedding(num_embeddings=num_unique_users,  embedding_dim=10)
        self.movies_embed = nn.Embedding(num_embeddings=num_unique_movies, embedding_dim=10)
        self.fc1 = nn.Linear(20, 120)
        self.fc2 = nn.Linear(120, 1)

    def forward(self, user_emb_idx, movie_emb_idx):
        # transfrom user_emb_idx and movie_emb_idx from (batch_size, 1) to (batch_size)
        user_emb_idx = user_emb_idx.squeeze(1)
        movie_emb_idx = movie_emb_idx.squeeze(1)
        
        user_emb = self.user_embed(user_emb_idx)
        movie_emb = self.movies_embed(movie_emb_idx)
        # конкатенируем эмбеддинги
        x = torch.cat((user_emb, movie_emb), dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # reshape output from (batch_size, 1) to (batch_size,)
        # x = x.squeeze(1)
        return x

'''
Initialize model on cuda and AdamW optimizer
with learning rate 0.001
'''
device = "cuda" if torch.cuda.is_available() else "cpu"


model = movielens_model(num_unique_users, num_unique_movies).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

'''
Function to calculate RMSE
'''
def rmse(pred, actual):
    return np.sqrt(np.mean(np.square(pred - actual)))

'''
Function to train model on cuda
takes dataloader, model, optimizer and number of epochs as input
and prints train and test RMSE after each epoch
'''
def train_model(train_loader, test_loader, model, optimizer, epochs):
    for epoch in range(epochs):
        # переводим модель в режим обучения
        model.train()
        # логируем train loss
        train_loss = 0

        # инициализируем функцию потерь
        criterion = nn.MSELoss(reduction ='mean')
        print(f"Epoch {epoch+1} of {epochs}")

        # итерируемся по пачке взаимодействий
        for i, (user_emb_idx, movie_emb_idx, ratings) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # переводим тензоры на cuda
            user_emb_idx = user_emb_idx.to(device)
            movie_emb_idx = movie_emb_idx.to(device)
            ratings = ratings.to(device)
            # преобразуем рейтинги из (batch_size, 1) в (batch_size, ) для корректной работы функции потерь
            ratings = ratings.view(-1, 1)
            # обнуляем градиенты
            optimizer.zero_grad()
            # получаем предсказания
            output = model(user_emb_idx, movie_emb_idx)
            # считаем функцию потерь
            loss = criterion(output, ratings)
            # обратный проход
            loss.backward()
            # обновляем веса
            optimizer.step()
            # логируем train loss
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Train loss: {train_loss}")
    
        model.eval()

        test_preds = []
        test_actual = []
        with torch.no_grad():
            for user_emb_idx, movie_emb_idx, ratings in test_loader:
                user_emb_idx = user_emb_idx.to(device)
                movie_emb_idx = movie_emb_idx.to(device)
                ratings = ratings.to(device)
                # преобразуем рейтинги из (batch_size, 1) в (batch_size, ) для корректной работы функции потерь
                ratings = ratings.view(-1, 1)
                output = model(user_emb_idx, movie_emb_idx)
                
                test_preds.append(output.detach().cpu().numpy())
                test_actual.append(ratings.detach().cpu().numpy())

            test_preds = np.concatenate(test_preds)
            test_actual = np.concatenate(test_actual)

            print(f"Test RMSE: {rmse(test_preds, test_actual)}")

'''
Run model for 10 epochs
'''
train_model(train_loader, test_loader, model, optimizer, 10)
