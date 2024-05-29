# Importa o módulo Counter da biblioteca collections para contar frequências
import pandas as pd
# Importa funções úteis do PyTorch para redes neurais e operações matemáticas
import torch.nn.functional as F
import torch.nn as nn
# Importa o módulo random para geração de valores aleatórios
import random
import pickle
# Importa o módulo torch para trabalhar com tensores e operações tensoriais
import torch
import numpy as np
# Importa o módulo os para interagir com o sistema operacional

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

def prepare_input(letras_a_divinhas, device, letras_indices_dict, acerto, modelo, letras, modelo_2):
    # Cria um vetor de zeros com tamanho 26 para representar as letras do alfabeto
    input_vector = torch.zeros(26, device=device)  # Inicializa um tensor de zeros no dispositivo especificado (CPU/GPU)
    
    # Função auxiliar para gerar um valor aleatório entre 0 e 100000000
    def random_value():
        return random.randint(0, 100000000)  
    
    print(modelo_2)  # Imprime o modelo 2 para depuração
    letras_a_divinhas_modelo2 = np.zeros(26)  # Inicializa um array numpy de zeros para armazenar as letras adivinhadas no modelo 2
    
    # Verifica se há alguma letra adivinhada
    if letras_a_divinhas == []:  # Se não há letras adivinhadas, prepara o modelo 2
        palavras = prepare_model2(letras)  # Chama a função prepare_model2 para preparar o modelo 2
        
    # Verifica qual modelo está sendo utilizado
    if modelo == 1:
        # Atualiza o vetor de entrada com base nas letras adivinhadas
        if letras_a_divinhas!= []:  # Se há letras adivinhadas
            for letra in letras:
                if letra in letras_a_divinhas:
                    if acerto:  # Se a letra foi adivinhada corretamente
                        value = random_value()  # Gera um valor aleatório
                        input_vector[letras_indices_dict[letra]] += value  # Atualiza o vetor de entrada
                    else:  # Se a letra não foi adivinhada corretamente
                        value = random_value()  # Gera um valor aleatório
                        input_vector[letras_indices_dict[letra]] -= value*len(letras_a_divinhas)  # Atualiza o vetor de entrada
                else:
                    value = random_value()  # Gera um valor aleatório
                    input_vector[letras_indices_dict[letra]] += value  # Atualiza o vetor de entrada
        # Imprime o vetor de entrada e retorna
        print(input_vector)
        return input_vector
    else:
        if letras_a_divinhas!= []:  # Se há letras adivinhadas
            for letra in letras:
                if letra in letras_a_divinhas:
                   letras_a_divinhas_modelo2[letras_indices_dict[letra]] = 1  # Marca a letra como adivinhada no modelo 2
            if acerto:
                vetores_filtrados = list(filter(lambda v: elementos_iguais(v, letras_a_divinhas_modelo2), modelo_2))  # Filtra palavras baseados nas letras adivinhadas acertadas
                return vetores_filtrados
            elif acerto == 0:
                vetores_filtrados = list(filter(lambda v: elementos_diferentes(v, letras_a_divinhas_modelo2), modelo_2))  # Filtra palavras baseados nas letras adivinhadas erradas
                return vetores_filtrados
        print("palavras")  # Imprime "palavras" para depuração
        return palavras  # Retorna as palavras preparadas
    
# Função auxiliar para verificar se dois vetores têm elementos iguais
def elementos_iguais(vetor1, vetor2):
    return sum((i == j and i!= 0 and j!= 0) for i, j in zip(vetor1, vetor2)) > 0

# Função auxiliar para verificar se dois vetores têm elementos diferentes
def elementos_diferentes(vetor1, vetor2):
    return sum((i == j and i!= 0 and j!= 0) for i, j in zip(vetor1, vetor2)) == 0

# Função para converter uma palavra em binário
def word_to_binary(palavra, letras):
    return [int(letra in palavra) for letra in letras]


def prepare_model2(letras):  # Define a função prepare_model2 que recebe uma lista de letras
    df = pd.read_excel("br-sem-acentos.xlsx", engine='openpyxl')  # Lê um arquivo Excel chamado "br-sem-acentos.xlsx" usando openpyxl como motor
    X = df["Palavra"].apply(lambda x: word_to_binary(str(x).lower(), letras)).tolist()  # Para cada palavra na coluna "Palavra" do DataFrame, converte-a para minúsculas, aplica a função word_to_binary para transformá-la em uma representação binária baseada nas letras fornecidas, e coleta todos esses vetores em uma lista
    return X  # Retorna a lista de vetores binários
    
# Função para obter uma lista de palavras de um arquivo
def get_palavras():
    caminho_do_arquivo = "/home/roger/Documentos/python/hangmania/dicionario/br-sem-acentos.txt"
    with open(caminho_do_arquivo, 'r') as arquivo:
        palavras = [palavra.strip().lower() for palavra in arquivo]
    return palavras

# Carrega um modelo pré-treinado
def carregar_modelo(device, model):
    if model == 1:
        model_path = "hangman_model_rnn.pt"
        model = RedeNeural(26, 26)  # Supondo que o modelo foi treinado com 26 entradas e saídas
        model.load_state_dict(torch.load(model_path))  # Carrega os pesos do modelo
        model.eval()  # Coloca o modelo em modo avaliação
        model.to(device)  # Mova o modelo para o dispositivo especificado (CPU/GPU)
    else:  # Esta é a cláusula else de uma estrutura condicional if...else
        filename = 'multi_output_logistic_Classifier.pkl'  # Define o nome do arquivo do modelo pré-treinado
        with open(filename, 'rb') as file:  # Abre o arquivo especificado em modo de leitura binária ('rb')
            model = pickle.load(file)  # Carrega o modelo pré-treinado do arquivo aberto usando a biblioteca pickle
    return model  # Retorna o modelo carregado para ser usado posteriormente


# Função principal que executa o jogo
def main():
    letras = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU se disponível, caso contrário CPU
    letras_indices_dict = {chr(i + ord('a')): i for i in range(26)}  # Dicionário para mapear letras para índices
    modelo = int(input("1 para rnn 0 para nlp "))
    model = carregar_modelo(device, modelo)  # Carrega o modelo
    palavras = get_palavras()  # Obtém uma lista de palavras
    palavra_secreta = random.choice(palavras)  # Seleciona uma palavra secreta aleatoriamente
    vidas = 6  # Número de tentativas permitidas
    palavra_desejada = "_" * len(palavra_secreta)  # Representação inicial da palavra secreta
    letras_a_divinhas = []  # Lista para armazenar as letras já adivinhadas
    while_letras_a_divinhas = [] # Lista para armazenar as letras já adivinhadas para o modelo 2
    acerto = 0
    modelo_2 = []
    
    # Loop principal do jogo
    while vidas >= 0:

        chute = prepare_input(letras_a_divinhas, device, letras_indices_dict, acerto, modelo, letras, modelo_2)  # Prepara a entrada para o modelo
        acerto = 0  # Reinicia o contador de acertos
        print("vidas:",vidas)  # Mostra o número de vidas restantes        
        print(palavra_desejada)  # Mostra a palavra secreta atualizada
        if modelo == 1: 
            with torch.no_grad():  # Desativa o cálculo de gradiente para a sessão de previsão
                output = model(chute.unsqueeze(0))  # Passa a entrada através do modelo
                _, predicted_letter_index = torch.max(output, dim=1)  # Encontra a letra prevista com maior probabilidade
                predicted_letter_index_value = predicted_letter_index.item()  # Converte o índice da letra prevista para um valor inteiro
            
            for letra, valor in letras_indices_dict.items():
                if valor == predicted_letter_index_value:
                    predicted_letter = letra  # Determina a letra prevista
        else:  # Esta é a cláusula else de uma estrutura condicional if...else
            modelo_2 = chute  # Assumindo que 'chute' seja uma variável que contém as letras previstas pelo usuário ou algum outro mecanismo
            print(len(modelo_2))  # Imprime o número de letras previstas
            letras_a_divinhas = []  # Inicializa uma lista vazia para armazenar as letras adivinhadas
            y_pred_probabilities = model.predict_proba(chute)  # Usa o modelo para calcular as probabilidades de cada letra ser a resposta correta
            predicted_letters = np.argmax(y_pred_probabilities, axis=1)  # Encontra o índice do máximo valor de probabilidade para cada letra, assumindo que isso corresponde à letra prevista
            print("predicted_letters", predicted_letters)  # Imprime as letras previstas
            for i in range(len(predicted_letters)):  # Itera sobre cada letra prevista
                print(i)  # Imprime o índice da letra prevista
                if predicted_letters[i][1] < len(predicted_letters) and predicted_letters[i][0] < len(predicted_letters):  # Verifica se ambas as letras estão dentro do intervalo válido
                    if letras[predicted_letters[i][0]] not in while_letras_a_divinhas:  # Verifica se a primeira letra prevista ainda não foi adicionada à lista de letras adivinhadas
                        predicted_letter = letras[predicted_letters[i][0]]  # Obtém a letra correspondente ao índice
                        while_letras_a_divinhas.append(predicted_letter)  # Adiciona a letra à lista de letras adivinhadas
                        break  # Sai do loop após adicionar a letra
                    if letras[predicted_letters[i][1]] not in while_letras_a_divinhas:  # Verifica se a segunda letra prevista ainda não foi adicionada à lista de letras adivinhadas
                        predicted_letter = letras[predicted_letters[i][1]]  # Obtém a letra correspondente ao índice
                        while_letras_a_divinhas.append(predicted_letter)  # Adiciona a letra à lista de letras adivinhadas
                        break  # Sai do loop após adicionar a letra
                elif predicted_letters[i][0] > len(predicted_letters):  # Caso especial para lidar com letras previstas fora do intervalo válido
                    predicted_letter = letras[predicted_letters[i][1]]  # Obtém a letra correspondente ao índice
                    if predicted_letter not in while_letras_a_divinhas:  # Verifica se a letra ainda não foi adicionada à lista de letras adivinhadas
                        while_letras_a_divinhas.append(predicted_letter)  # Adiciona a letra à lista de letras adivinhadas
                        break  # Sai do loop após adicionar a letra
                elif predicted_letters[i][1] > len(predicted_letters):  # Outro caso especial para lidar com letras previstas fora do intervalo válido
                    predicted_letter = letras[predicted_letters[i][0]]  # Obtém a letra correspondente ao índice
                    if predicted_letter not in while_letras_a_divinhas:  # Verifica se a letra ainda não foi adicionada à lista de letras adivinhadas
                        while_letras_a_divinhas.append(predicted_letter)  # Adiciona a letra à lista de letras adivinhadas
                        break  # Sai do loop após adicionar a letra
          
        print("letra prevista",predicted_letter)
            
        if predicted_letter in palavra_secreta:
            for i in range(len(palavra_secreta)):
                if palavra_secreta[i] == predicted_letter:
                    acerto += 1  # Incrementa o contador de acertos
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
