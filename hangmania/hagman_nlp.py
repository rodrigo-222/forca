# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Função para converter palavras em características binárias
def word_to_binary(word, alphabet):
    return [int(letter in word) for letter in alphabet]

# Ler o arquivo Excel contendo as palavras sem acentos
df = pd.read_excel("br-sem-acentos.xlsx", engine='openpyxl')

# Definir o alfabeto completo
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Converter palavras em características binárias usando a função word_to_binary
X = df.iloc[:, 0].apply(lambda x: word_to_binary(x.lower(), alphabet)).tolist()

# Criar os rótulos com base nas frequências de cada letra nas palavras
num_rows = len(df)
y = np.zeros((num_rows, 26))  # Inicializa uma matriz de zeros com dimensões adequadas
for index, row in df.iterrows():
    for letter in alphabet:
        if letter in row['Palavra'].lower():  # Verifica se a letra está presente na palavra
            y[index, alphabet.index(letter)] = 1  # Marca a coluna correspondente como 1

# Imprimir o vetor de características binárias X para verificação
print(X)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo separado para cada saída usando Random Forest Classifier
models = []
for i in range(y_train.shape[1]):  # Loop através de cada saída
    model = RandomForestClassifier(random_state=42)  # Instancia um modelo de classificação aleatória
    model.fit(X_train, y_train[:, i])  # Treina o modelo com os dados de treinamento
    models.append(model)  # Armazena o modelo treinado

# Envolver os modelos individuais com MultiOutputClassifier para suporte a múltiplas saídas
multi_output_classifier = MultiOutputClassifier(RandomForestClassifier(random_state=42), n_jobs=-1)
multi_output_classifier.fit(X_train, y_train)  # Treina o classificador de múltiplas saídas

# Fazer previsões no conjunto de teste
y_pred_probabilities = multi_output_classifier.predict_proba(X_test)

# Identificar a letra mais provável para cada palavra
predicted_letters = np.argmax(y_pred_probabilities, axis=1)
print(predicted_letters)
print(len(predicted_letters))

# Iterar sobre cada palavra do conjunto de teste e imprimir a letra mais provável
for i in range(len(predicted_letters)):
    print(f"Palavra: {X_test[i]}, Letra mais provável: {alphabet[predicted_letters[i][0]]}")
