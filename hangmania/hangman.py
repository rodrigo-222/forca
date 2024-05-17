# Importa o módulo Counter da biblioteca collections para contar frequências
from collections import Counter
# Importa funções úteis do PyTorch para redes neurais e operações matemáticas
import torch.nn.functional as F
import torch.nn as nn
# Importa o módulo random para geração de valores aleatórios
import random
# Importa o módulo torch para trabalhar com tensores e operações tensoriais
import torch
# Importa o módulo os para interagir com o sistema operacional
import os

# Define uma classe para a rede neural simples
class RedeNeural(nn.Module):
    def __init__(self, entrada, saída):
        # Inicializa a rede neural com duas camadas totalmente conectadas (fully connected layers)
        super(RedeNeural, self).__init__()
        self.fc1 = nn.Linear(entrada, 128)  # Camada oculta com 128 neurônios
        self.fc2 = nn.Linear(128, saída)   # Camada de saída com número igual ao número de classes/saídas

    def forward(self, x):
        # Define como os dados passam pela rede durante a propagação
        x = self.fc1(x)       # Passa pelo primeiro layer linear
        x = F.relu(x)         # Aplica a função ReLU (Rectified Linear Unit)
        x = self.fc2(x)       # Passa pelo segundo layer linear
        return x              # Retorna a saída final

# Função para preparar a entrada para o modelo
def prepare_input(letras_a_divinhas, device, letras_indices_dict, acerto):
    # Cria um vetor de zeros com tamanho 26 para representar as letras do alfabeto
    input_vector = torch.zeros(26, device=device)
    # Lista com todas as letras do alfabeto
    letras = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    
    # Função auxiliar para gerar um valor aleatório
    def random_value():
        return random.randint(0, 100000000)  # Alterado para gerar apenas valores positivos
    
    # Atualiza o vetor de entrada com base nas letras adivinhadas
    if letras_a_divinhas != []:
        for letra in letras:
            if letra in letras_a_divinhas:
                if acerto:  # Se a letra foi adivinhada corretamente
                    value = random_value()
                    input_vector[letras_indices_dict[letra]] += value
                else:  # Se a letra não foi adivinhada corretamente
                    value = random_value()
                    input_vector[letras_indices_dict[letra]] -= value*100
            else:
                value = random_value()
                input_vector[letras_indices_dict[letra]] += value 
        # Imprime o vetor de entrada e retorna
    print(input_vector)
    return input_vector

# Função para obter uma lista de palavras de um arquivo
def get_palavras():
    caminho_do_arquivo = "/home/roger/Documentos/python/hangmania/dicionario/br-sem-acentos.txt"
    with open(caminho_do_arquivo, 'r') as arquivo:
        palavras = [palavra.strip().lower() for palavra in arquivo]
    return palavras

# Função para calcular a frequência de cada letra em todas as palavras
def get_frequencia_palavras():
    palavras = get_palavras()
    letras = ''.join(palavras)
    frequencia = Counter(letras)
    return frequencia

# Carrega um modelo pré-treinado
def carregar_modelo(device):
    model_path = "hangman_model_rnn.pt"
    model = RedeNeural(26, 26)  # Supondo que o modelo foi treinado com 26 entradas e saídas
    model.load_state_dict(torch.load(model_path))  # Carrega os pesos do modelo
    model.eval()  # Coloca o modelo em modo avaliação
    model.to(device)  # Mova o modelo para o dispositivo especificado (CPU/GPU)
    return model

# Função principal que executa o jogo
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU se disponível, caso contrário CPU
    letras_indices_dict = {chr(i + ord('a')): i for i in range(26)}  # Dicionário para mapear letras para índices
    model = carregar_modelo(device)  # Carrega o modelo
    frequencia_letras = get_frequencia_palavras()  # Obtém a frequência de letras
    palavras = get_palavras()  # Obtém uma lista de palavras
    palavra_secreta = random.choice(palavras)  # Seleciona uma palavra secreta aleatoriamente
    entrada = 26  # Número de entradas para o modelo (26 letras do alfabeto)
    letras_tens = torch.zeros(len(frequencia_letras), entrada).to(device)  # Vetor de entrada inicializado com zeros
    vidas = 6  # Número de tentativas permitidas
    palavra_desejada = "_" * len(palavra_secreta)  # Representação inicial da palavra secreta
    letras_a_divinhas = []  # Lista para armazenar as letras já adivinhadas
    acerto = 0  # Contador de acertos
    
    # Loop principal do jogo
    while vidas >= 0:
        input("pausa")  # Espera por uma entrada do usuário antes de continuar
        chute = prepare_input(letras_a_divinhas, device, letras_indices_dict, acerto)  # Prepara a entrada para o modelo
        acerto = 0  # Reinicia o contador de acertos
        print(vidas)  # Mostra o número de vidas restantes
        print(palavra_desejada)  # Mostra a palavra secreta atualizada
        
        with torch.no_grad():  # Desativa o cálculo de gradiente para a sessão de previsão
            output = model(chute.unsqueeze(0))  # Passa a entrada através do modelo
            _, predicted_letter_index = torch.max(output, dim=1)  # Encontra a letra prevista com maior probabilidade
            predicted_letter_index_value = predicted_letter_index.item()  # Converte o índice da letra prevista para um valor inteiro
        
        for letra, valor in letras_indices_dict.items():
            if valor == predicted_letter_index_value:
                predicted_letter = letra  # Determina a letra prevista
        print(predicted_letter)  # Mostra a letra prevista
        
        if predicted_letter in palavra_secreta:
            for i in range(len(palavra_secreta)):
                if palavra_secreta[i] == predicted_letter:
                    acerto += 1  # Incrementa o contador de acertos
                    print("entro", acerto)
                    palavra_desejada = palavra_desejada[:i] + predicted_letter + palavra_desejada[i+1:]  # Atualiza a palavra secreta
        else:
            if predicted_letter not in letras_a_divinhas:
                vidas -= 1  # Decrementa o número de vidas se a letra não estiver na palavra secreta
        
        letras_a_divinhas.append(predicted_letter)  # Adiciona a letra prevista à lista de letras adivinhadas
        if "_" not in palavra_desejada:
            print(palavra_desejada)
            print("Parabéns Você ganhou!")
            break  # Termina o loop se a palavra secreta for completamente revelada
        
    if vidas <= 0:
        print(palavra_secreta)  # Mostra a palavra secreta se o jogador perdeu todas as suas vidas

# Executa a função main para iniciar o jogo
if __name__ == "__main__":
    main()
