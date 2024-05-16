import os  # Importa o módulo os para operações de sistema, como manipulação de arquivos
import torch  # Importa o módulo torch para trabalhar com Tensors e redes neurais
import torch.nn as nn  # Importa o módulo nn para construir redes neurais
import torch.optim as optim  # Importa o módulo optim para otimizadores de aprendizado de máquina
from collections import Counter  # Importa o Counter para contar frequências

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Configura o dispositivo para uso de GPU se disponível, caso contrário, CPU

letras_indices_dict = {chr(i + ord('a')): i for i in range(26)}  # Mapeia letras minúsculas do alfabeto para índices numéricos
print(letras_indices_dict)

def get_frequencia_palavras():  # Função para obter a frequência de palavras
    caminho_do_arquivo = os.path.join("/home/roger/Documentos/python/hangmania/dicionario", "br-sem-acentos.txt")  # Caminho do arquivo de palavras
    with open(caminho_do_arquivo, 'r') as arquivo:  # Abre o arquivo em modo leitura
        palavras = [palavra.strip() for palavra in arquivo]  # Lê cada linha do arquivo e remove espaços em branco
        letras_counter = []  # Lista temporária para armazenar palavras convertidas em letras
        for palavra in palavras:  # Itera sobre cada palavra lida
            palavra = palavra.lower()  # Converte a palavra para minúscula
            letras_counter.append(palavra)  # Adiciona a palavra convertida em letras à lista
        letras = ''.join(letras_counter)  # Junta todas as letras em uma única string
        frequencia = Counter(letras)  # Conta a frequência de cada letra
        return frequencia  # Retorna o objeto Counter com as frequências

class RedeNeural(nn.Module):  # Classe para definir a estrutura da rede neural
    def __init__(self, entrada, saída):  # Construtor da classe
        super(RedeNeural, self).__init__()  # Chama o construtor da classe pai
        self.fc1 = nn.Linear(entrada, 128)  # Primeira camada linear
        self.fc2 = nn.Linear(128, saída)  # Segunda camada linear

    def forward(self, x):  # Método para definir o fluxo de dados através da rede
        x = self.fc1(x)  # Passa os dados pela primeira camada linear
        x = torch.relu(x)  # Aplica a função de ativação ReLU
        x = self.fc2(x)  # Passa os dados pela segunda camada linear
        return x  # Retorna os resultados finais

frequencia_letras = get_frequencia_palavras()  # Obtém a frequência de letras do arquivo
entrada = 26  # Número de entradas (letras do alfabeto)
saída = 26  # Número de saídas (também letras do alfabeto)
rede = RedeNeural(entrada, saída).to(device)  # Cria a rede neural e move para o dispositivo configurado
criterion = nn.CrossEntropyLoss().to(device)  # Define o critério de perda como CrossEntropyLoss
optimizer = optim.Adam(rede.parameters(), lr=0.001)  # Define o otimizador Adam com taxa de aprendizado 0.001

# Preparando os dados de entrada
letras_tens = torch.zeros(len(frequencia_letras), entrada).to(device)  # Cria um tensor de zeros com dimensões adequadas
print(letras_tens)  # Imprime o tensor de entrada para depuração
for palavra, frequencia in frequencia_letras.items():  # Itera sobre cada letra e sua frequência
    indice = letras_indices_dict[palavra]  # Obtem o índice da letra no dicionário
    letras_tens[indice] = frequencia  # Atualiza o valor no tensor de entrada com a frequência da letra
    print(indice)

print(letras_tens)
# Preparando o tensor de alvo
indices = torch.tensor([letras_indices_dict[letra] for letra in frequencia_letras.keys()], dtype=torch.long).to(device)  # Cria o tensor de alvos
for letra in frequencia_letras.keys():
    print(letra)
    print(letras_indices_dict[letra])
print(indices)
for epoch in range(100000):  # Loop de treinamento
    optimizer.zero_grad()  # Zera os gradientes
    output = rede(letras_tens)  # Faz a passagem dos dados pela rede
    #print(output)
    loss = criterion(output, indices)  # Calcula a perda
    loss.backward()  # Realiza o backpropagation
    optimizer.step()  # Atualiza os parâmetros da rede

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')  # Imprime a perda

torch.save(rede.state_dict(), "hangman_model.pt")  # Salva o estado da rede neural
