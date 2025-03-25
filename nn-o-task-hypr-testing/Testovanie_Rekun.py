

import kagglehub
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import random
from torch import optim
from sklearn.metrics import r2_score, explained_variance_score
import os


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False



#  Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss  

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ModelRegv0(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(ModelRegv0, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim + 10),
            nn.ReLU(),
            nn.Linear(self.input_dim + 10, self.input_dim + 10),
            nn.ReLU(),
            nn.Linear(self.input_dim + 10, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim)
        )
        
    def forward(self,x):
        out = self.layers(x)
        return out

class ModelRegv1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2 + 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.input_dim * 2 + 2, self.input_dim + 1),
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim + 1),
            nn.Linear(self.input_dim + 1, (self.input_dim + 1) // 2),
            nn.ReLU(),
            nn.Linear((self.input_dim + 1) // 2, self.output_dim)
        )
        
    def forward(self, x):
        out = self.layers(x)
        return out

class ModelRegv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2),  # 31 -> 62
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(self.input_dim * 2, self.input_dim),  # 62 -> 31
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim),
            nn.Dropout(0.3),
            nn.Linear(self.input_dim, self.input_dim // 2),  # 31 -> 15
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim)  # 15 -> 1
        )
        
    def forward(self, x):
        out = self.layers(x)
        return out

class ModelRegv3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv3, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 3),  
            nn.ReLU(),
            nn.Linear(self.input_dim * 3, self.input_dim * 2),  
            nn.ReLU(),
            nn.Linear(self.input_dim * 2, self.input_dim + 32),  
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(self.input_dim + 32, self.input_dim // 2 + 16),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2 + 16, self.input_dim // 4 + 8),
            nn.ReLU(),
            nn.Linear(self.input_dim // 4 + 8, self.output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class ModelRegv4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv4, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim + 20),
            nn.ReLU(),
            nn.Linear(self.input_dim + 20, self.input_dim + 10),
            nn.ReLU(),
            nn.Linear(self.input_dim + 10, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.input_dim // 4),
            nn.ReLU(),
            nn.Linear(self.input_dim // 4, self.output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)



class ModelRegv4_t(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv4_t, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim + 20),
            nn.Tanh(),
            nn.Linear(self.input_dim + 20, self.input_dim + 10),
            nn.Tanh(),
            nn.Linear(self.input_dim + 10, self.input_dim),
            nn.Tanh(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.Tanh(),
            nn.Linear(self.input_dim // 2, self.input_dim // 4),
            nn.Tanh(),
            nn.Linear(self.input_dim // 4, self.output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


class ModelRegv4_lr(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv4_lr, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim + 20),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim + 20, self.input_dim + 10),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim + 10, self.input_dim),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim // 2, self.input_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim // 4, self.output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


class ModelRegv4_el(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv4_el, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim + 20),
            nn.ELU(),
            nn.Linear(self.input_dim + 20, self.input_dim + 10),
            nn.ELU(),
            nn.Linear(self.input_dim + 10, self.input_dim),
            nn.ELU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ELU(),
            nn.Linear(self.input_dim // 2, self.input_dim // 4),
            nn.ELU(),
            nn.Linear(self.input_dim // 4, self.output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


class ModelRegv4_sl(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv4_sl, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim + 20),
            nn.SELU(),
            nn.Linear(self.input_dim + 20, self.input_dim + 10),
            nn.SELU(),
            nn.Linear(self.input_dim + 10, self.input_dim),
            nn.SELU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.SELU(),
            nn.Linear(self.input_dim // 2, self.input_dim // 4),
            nn.SELU(),
            nn.Linear(self.input_dim // 4, self.output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)



#data
store_df = pd.read_csv('store.csv')
train_df = pd.read_csv('train.csv')

store_df['CompetitionDistance']= store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].median())
store_df['CompetitionOpenSinceMonth'] = store_df['CompetitionOpenSinceMonth'].fillna(0)
store_df['CompetitionOpenSinceYear'] = store_df['CompetitionOpenSinceYear'].fillna(0)
store_df['HasCompetition'] = store_df['CompetitionOpenSinceMonth'] != 0
store_df['Promo2SinceWeek'] = store_df['Promo2SinceWeek'].fillna(0)
store_df['Promo2SinceYear'] = store_df['Promo2SinceYear'].fillna(0)
store_df['PromoInterval'] = store_df['PromoInterval'].fillna('None')

store_df = pd.get_dummies(store_df, columns=['StoreType', 'Assortment', 'PromoInterval'], prefix = ['StoreType', 'Assortment', 'PromoInterval'], dtype=int)
store_df['HasCompetition'] = store_df['HasCompetition'].astype(int)


fig,ax = plt.subplots(figsize = (20,20))
sns.heatmap(store_df.corr(), ax = ax, annot=True)
plt.savefig('store_df_corr.png')


fig,ax = plt.subplots(figsize = (20,20))
sns.heatmap(train_df.corr(numeric_only=True), ax = ax, annot=True)

plt.savefig('train_df_corr.png')

train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df = train_df.sort_values(by='Date')
train_df['StateHoliday'] = train_df['StateHoliday'].astype(str)
train_df = pd.get_dummies(train_df, columns=['StateHoliday'], prefix=['StateHoliday'], dtype=int)

train_df['Year'] = train_df['Date'].dt.year.astype('int64')
train_df['Month'] = train_df['Date'].dt.month.astype('int64')
train_df['Day'] = train_df['Date'].dt.day.astype('int64')

merged_df = pd.merge(train_df, store_df, on='Store', how='left')

#EDA

fig,ax = plt.subplots(figsize = (40,20))
sns.heatmap(merged_df.corr(), ax = ax, annot=True)
plt.savefig('merged_df_corr.png')

# Sales
plt.figure(figsize=(10, 5))
sns.histplot(merged_df['Sales'], kde=True)
plt.title(' Sales distribution')
plt.savefig('Sales_distribution.png')

merged_df.set_index('Date', inplace=True)
plt.figure(figsize=(14, 6))
merged_df['Sales'].resample('W').sum().plot()
plt.title('Sales by week')
plt.xlabel('Data')
plt.ylabel('Sales')
plt.savefig('Sales_by_week.png')

train_df = train_df.drop('Date', axis=1)

plt.figure(figsize=(8, 4))
sns.boxplot(x='Promo', y='Sales', data=merged_df)
plt.title('Impact of Promo on Sales')
plt.savefig('Promo_on_Sales.png')


plt.figure(figsize=(8, 4))
sns.boxplot(x='DayOfWeek', y='Sales', data=merged_df.reset_index())
plt.title('The Impact of Day of the Week on Sales')
plt.savefig('Day_on_Sales.png')

plt.show()

length = merged_df.shape[0]
train_len = int(length * 0.7)
test_len = int(length * 0.15)
val_len = length - train_len - test_len

print(length - train_len - test_len - val_len)

train_df = merged_df.iloc[0:train_len]
test_df = merged_df.iloc[train_len:train_len + test_len]
val_df = merged_df.iloc[train_len + test_len:]
print(f'train_df {train_df.shape}\ntest_df {test_df.shape}\nval_df {val_df.shape}')

y_train = train_df['Sales']
X_train = train_df.drop('Sales', axis=1)
y_test = test_df['Sales']
X_test = test_df.drop('Sales', axis=1)
y_val = val_df['Sales']
X_val = val_df.drop('Sales', axis=1)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
X_val = torch.tensor(X_val,dtype=torch.float32)

y_train = torch.tensor(y_train.to_numpy(),dtype=torch.float32).reshape(-1,1)
y_test = torch.tensor(y_test.to_numpy(),dtype=torch.float32).reshape(-1,1)
y_val = torch.tensor(y_val.to_numpy(),dtype=torch.float32).reshape(-1,1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)




merged_df.describe().to_csv("merged_df_description.csv")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
max_epochs = 60
learning_rate = 0.01
model = ModelRegv4_t(31, 1)  
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#  EarlyStopping
early_stopping = EarlyStopping(patience=10, verbose=True, path='best_model.pt')

train_loss_metr = []
val_loss_metr = []
train_r2_metr = []
val_r2_metr = []
train_exp_var_metr = []
val_exp_var_metr = []

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    train_outputs = []
    train_targets = []
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        
        train_outputs.append(output.cpu().detach().numpy())
        train_targets.append(target.cpu().detach().numpy())
    
    train_loss /= len(train_loader.dataset)
    
    train_outputs = np.concatenate(train_outputs)
    train_targets = np.concatenate(train_targets)
    train_r2 = r2_score(train_targets, train_outputs)
    train_exp_var = explained_variance_score(train_targets, train_outputs)
    
    model.eval()
    val_loss = 0.0
    val_outputs = []
    val_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
            
            val_outputs.append(output.cpu().detach().numpy())
            val_targets.append(target.cpu().detach().numpy())
            
        val_loss /= len(val_loader.dataset)
        
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)
        val_r2 = r2_score(val_targets, val_outputs)
        val_exp_var = explained_variance_score(val_targets, val_outputs)
    
    train_loss_metr.append(train_loss)
    val_loss_metr.append(val_loss)
    train_r2_metr.append(train_r2)
    val_r2_metr.append(val_r2)
    train_exp_var_metr.append(train_exp_var)
    val_exp_var_metr.append(val_exp_var)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, "
          f"Train ExpVar: {train_exp_var:.4f}, Val ExpVar: {val_exp_var:.4f}")
    print('\n')

    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

model.load_state_dict(torch.load('best_model.pt'))
print("Loaded the best model based on validation loss.")

# visualisation
epochs = range(1, len(train_loss_metr) + 1)

plt.figure(figsize=(15, 10))

#  Loss
plt.subplot(3, 1, 1)
plt.plot(epochs, train_loss_metr, label='Train Loss', marker='o')
plt.plot(epochs, val_loss_metr, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.xticks(range(1, len(train_loss_metr) + 1, 5))  
plt.yticks(np.linspace(min(min(train_loss_metr), min(val_loss_metr)), 
                       max(max(train_loss_metr), max(val_loss_metr)), 10)) 
plt.title('Train and Validation Loss')
plt.legend()
plt.grid(True)

#  R²
plt.subplot(3, 1, 2)
plt.plot(epochs, train_r2_metr, label='Train R²', marker='o')
plt.plot(epochs, val_r2_metr, label='Validation R²', marker='o')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.xticks(range(1, len(train_loss_metr) + 1, 5)) 
plt.yticks(np.linspace(min(min(train_r2_metr), min(val_r2_metr)), 
                       max(max(train_r2_metr), max(val_r2_metr)), 10))  
plt.title('Train and Validation R²')
plt.legend()
plt.grid(True)

# Explained Variance
plt.subplot(3, 1, 3)
plt.plot(epochs, train_exp_var_metr, label='Train Explained Variance', marker='o')
plt.plot(epochs, val_exp_var_metr, label='Validation Explained Variance', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Explained Variance Score')
plt.xticks(range(1, len(train_loss_metr) + 1, 5))  
plt.yticks(np.linspace(min(min(train_exp_var_metr), min(val_exp_var_metr)), 
                       max(max(train_exp_var_metr), max(val_exp_var_metr)), 10))  
plt.title('Train and Validation Explained Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('metrics_final.png')
plt.show()

model.load_state_dict(torch.load('best_model.pt'))
model = model.to(device)
model.eval()

test_loss = 0.0
test_outputs = []
test_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)  
        output = model(data) 
        loss = criterion(output, target)  #  loss
        test_loss += loss.item() * data.size(0)  
        
        test_outputs.append(output.cpu().numpy())
        test_targets.append(target.cpu().numpy())

test_loss /= len(test_loader.dataset)

test_outputs = np.concatenate(test_outputs)
test_targets = np.concatenate(test_targets)

test_r2 = r2_score(test_targets, test_outputs)
test_exp_var = explained_variance_score(test_targets, test_outputs)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test Explained Variance: {test_exp_var:.4f}")

results_path = os.path.join(os.path.dirname('best_model.pt'), 'test_results1.txt')
with open(results_path, 'w') as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test R²: {test_r2:.4f}\n")
    f.write(f"Test Explained Variance: {test_exp_var:.4f}\n")

print(f"{results_path}")
