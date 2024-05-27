import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import pickle

# Função para converter palavras em características binárias
def word_to_binary(word, alphabet):
    return [int(letter in word) for letter in alphabet]

# Ler o arquivo Excel
df = pd.read_excel("br-sem-acentos.xlsx", engine='openpyxl')

# Definir o alfabeto
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"] # Lista de todas as letras do alfabeto

# Converter palavras em características binárias
X = df.iloc[:, 0].apply(lambda x: word_to_binary(x.lower(), alphabet)).tolist()

# Criar os rótulos com base nas frequências de cada letra nas palavras
num_rows = len(df)
print(num_rows)
y = np.zeros((num_rows, 26))  # Matriz de zeros com num_rows linhas e 26 colunas
for index, row in df.iterrows():
    for letter in alphabet:
        if letter in row['Palavra'].lower():  # Verifica se a letra está na palavra
            y[index, alphabet.index(letter)] = 1  # Marca a coluna correspondente como 1

print(X)
# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo para cada saída usando LogisticRegression
models = []
for i in range(y_train.shape[1]):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train[:, i])
    models.append(model)

# Envolver os modelos com MultiOutputClassifier
multi_output_classifier = MultiOutputClassifier(RandomForestClassifier( random_state=42), n_jobs=-1)
multi_output_classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste e encontrar a letra mais provável para cada palavra
y_pred_probabilities = multi_output_classifier.predict_proba(X_test)

predicted_letters = np.argmax(y_pred_probabilities, axis=1)
print(predicted_letters)
print(len(predicted_letters))
# Iterar sobre cada linha do resultado do np.argmax para extrair a letra correspondente do alfabeto
for i in range(len(predicted_letters)):
    print(f"Palavra: {X_test[i]}, Letra mais provável: {alphabet[predicted_letters[i][0]]}")

for i in range(len(predicted_letters)):
    print(f"Palavra: {X_test[i]}, Letra mais provável: {alphabet[predicted_letters[i][1]]}")
filename = 'multi_output_logistic_Classifier.pkl'
with open(filename, 'wb') as file:
    pickle.dump(multi_output_classifier, file)
