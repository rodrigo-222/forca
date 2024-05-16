import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class HangmanDataset(Dataset):
    def __init__(self, encoded_words, targets):
        self.encoded_words = encoded_words
        self.targets = targets

    def __len__(self):
        return len(self.encoded_words)

    def __getitem__(self, idx):
        return self.encoded_words[idx], self.targets[idx]

class HangmanModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(HangmanModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        x = self.fc(last_hidden_state)
        return x

def load_and_prepare_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '').lower()

    char_to_index = {char: i for i, char in enumerate(sorted(set(data)))}
    index_to_char = {i: char for i, char in enumerate(sorted(set(data)))}

    def encode_word(word):
        return [char_to_index[char] for char in word]

    encoded_words = [encode_word(word) for word in data.split()]
    targets = [word + [0] for word in encoded_words]  # Adiciona um token final para indicar o fim da palavra

    return encoded_words, targets, char_to_index, index_to_char

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0).to(device)  # Adiciona uma dimensão extra para batch
        targets = torch.tensor(targets, dtype=torch.long).unsqueeze(0).to(device)  # Adiciona uma dimensão extra para batch
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # Corrige a forma do tensor de saída para corresponder à forma do tensor de destino
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0).to(device)  # Adiciona uma dimensão extra para batch
            targets = torch.tensor(targets, dtype=torch.long).unsqueeze(0).to(device)  # Adiciona uma dimensão extra para batch
            
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

device = torch.device('cpu')
file_path = '/home/roger/Documentos/python/hangmania/dicionario/br-sem-acentos.txt'
encoded_words, targets, char_to_index, index_to_char = load_and_prepare_data(file_path)

vocab_size = len(char_to_index)
max_seq_len = max([len(seq) for seq in encoded_words])

dataset = HangmanDataset(encoded_words, targets)
dataloader_train = DataLoader(dataset, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset, batch_size=64, shuffle=False)

model = HangmanModel(vocab_size, embedding_dim=50, hidden_dim=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(model, dataloader_train, criterion, optimizer, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}')

test_loss = evaluate_model(model, dataloader_test, criterion, device)
print(f'Test Loss: {test_loss}')
