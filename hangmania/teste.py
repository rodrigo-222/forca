import torch
import torch.nn as nn
import torch.nn.functional as F

# Definição da classe RedeNeural
class RedeNeural(nn.Module):
    def __init__(self, entrada, saída):
        super(RedeNeural, self).__init__()
        self.fc1 = nn.Linear(entrada, 128)
        self.fc2 = nn.Linear(128, saída)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Carregar o modelo treinado
model_path = "hangman_model.pt"

# Criar uma instância da classe RedeNeural
model = RedeNeural(26, 26)  # Supondo que o modelo foi treinado com 26 entradas e saídas

# Carregar o estado do modelo treinado na instância
model.load_state_dict(torch.load(model_path))

# Colocar o modelo em modo de avaliação
model.eval()

# Especificar o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mover o modelo para o dispositivo especificado
model.to(device)

def predict_letter(sequence):
    """
    Usa o modelo para prever qual letra pode estar faltando na sequência.
    """
    # Converter a sequência de letras em um tensor e mover para o dispositivo especificado
    sequence_tensor = torch.tensor([ord(char) - ord('a') for char in sequence], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Fazer a passagem do tensor pela rede neural
    output = model(sequence_tensor)
    
    # Encontrar a letra com a maior probabilidade
    _, predicted = torch.max(output.data, 1)
    
    # Converter o índice da letra de volta para letra e retornar
    return chr(predicted.item() + ord('a'))

# Solicita ao usuário que insira uma sequência de letras
sequence = "abcdefghijklmnopqrstuvwxyz"

# Faz a previsão
while(True):
    predicted_letter = predict_letter(sequence)

    print(f"A letra prevista é: {predicted_letter}")
