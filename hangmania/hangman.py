from collections import Counter
import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import os

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

def prepare_input(letras_a_divinhas, device,letras_indices_dict, acerto):
    # Cria um vetor de zeros com tamanho 26
    input_vector = torch.zeros(26, device=device)
    # Marca as posições das letras adivinhadas
    
    for letra in letras_a_divinhas:
        if acerto == 0:
            input_vector[letras_indices_dict[letra]] -= 100000^acerto
        else:
            input_vector[letras_indices_dict[letra]] += 100000^acerto

    print(input_vector)
    return input_vector

def get_palavras():
    caminho_do_arquivo = "/home/roger/Documentos/python/hangmania/dicionario/br-sem-acentos.txt"
    with open(caminho_do_arquivo, 'r') as arquivo:
        palavras = [palavra.strip().lower() for palavra in arquivo]
    return palavras

def get_frequencia_palavras():
    palavras = get_palavras()
    letras = ''.join(palavras)
    frequencia = Counter(letras)
    return frequencia

def carregar_modelo(device):
    model_path = "hangman_model.pt"
    model = RedeNeural(26, 26)  # Supondo que o modelo foi treinado com 26 entradas e saídas
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Modo avaliação
    model.to(device)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    letras_indices_dict = {chr(i + ord('a')): i for i in range(26)}
    model = carregar_modelo(device)
    frequencia_letras = get_frequencia_palavras()
    palavras = get_palavras()
    palavra_secreta = random.choice(palavras)
    entrada = 26
    letras_tens = torch.zeros(len(frequencia_letras), entrada).to(device)
    vidas = 6
    palavra_desejada = "_" * len(palavra_secreta)
    letras_a_divinhas = []
    acerto = 0
    while vidas >= 0:
        input("pausa")
        chute = prepare_input(letras_a_divinhas, device, letras_indices_dict, acerto)
        acerto = 0
        print(vidas)
        print(palavra_desejada)


        with torch.no_grad():
            output = model(chute.unsqueeze(0))
            _, predicted_letter_index = torch.max(output, dim=1)
            predicted_letter_index_value = predicted_letter_index.item()
        

        for letra, valor in letras_indices_dict.items():
            if valor == predicted_letter_index_value:
                predicted_letter = letra
        letras_a_divinhas.append(predicted_letter)
        print(predicted_letter)
        if predicted_letter in palavra_secreta:
            for i in range(len(palavra_secreta)):
                if palavra_secreta[i] == predicted_letter:
                    acerto=+1 
                    print("entro", acerto)
                    palavra_desejada = palavra_desejada[:i] + predicted_letter + palavra_desejada[i+1:]
        else:
            if predicted_letter not in letras_a_divinhas:
                print(letras_a_divinhas)
                print(predicted_letter)
                vidas -= 1
        
        if "_" not in palavra_desejada:
            print(palavra_desejada)
            print("Parabéns Você ganhou!")
            break
    if vidas == 0:
        print(palavra_secreta)

main()