import os
import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class PhishingDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path, dtype=np.float32, header=0)
        scaler = MinMaxScaler()
        self.df['num_words_norm'] = scaler.fit_transform(self.df[['num_words']])
        self.df['num_unique_words_norm'] = scaler.fit_transform(self.df[['num_unique_words']])
        self.df['num_stopwords_norm'] = scaler.fit_transform(self.df[['num_words']])
        self.df['num_links_norm'] = scaler.fit_transform(self.df[['num_links']])
        self.df['num_unique_domains_norm'] = scaler.fit_transform(self.df[['num_unique_domains']])
        self.df['num_email_addresses_norm'] = scaler.fit_transform(self.df[['num_email_addresses']])
        self.df['num_spelling_errors_norm'] = scaler.fit_transform(self.df[['num_spelling_errors']])
        self.df['num_urgent_keywords_norm'] = scaler.fit_transform(self.df[['num_urgent_keywords']])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ten = torch.tensor(
            [self.df['num_words_norm'][idx], self.df['num_unique_words_norm'][idx], self.df['num_stopwords_norm'][idx],
             self.df['num_links_norm'][idx],
             self.df['num_unique_domains_norm'][idx], self.df['num_email_addresses_norm'][idx],
             self.df['num_spelling_errors_norm'][idx],
             self.df['num_urgent_keywords_norm'][idx]])
        return ten, self.df['label'][idx]


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)
        self.r3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        x = self.r3(x)
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    model = NeuralNet()
    dataset = PhishingDataset('train.csv')
    dataloader = DataLoader(batch_size=10, dataset=dataset, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    torch.save(model.state_dict(), 'weights.pt')
