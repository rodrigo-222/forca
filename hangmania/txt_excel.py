import pandas as pd

# Função para ler o arquivo e contar as ocorrências das letras
def processar_palavras(arquivo):
    # Ler o arquivo linha por linha
    with open(arquivo, 'r') as file:
        palavras = file.read().splitlines()
    
    # Inicializar um dicionário para armazenar contagens de letras
    contagem_letras = {chr(65 + i): [''] * len(palavras) for i in range(26)}
    
    # Processar cada palavra
    for i,palavra in enumerate(palavras):
        print(i)
        for letra in palavra:
            if letra.isalpha():  # Verificar se é uma letra
                indice = ord(letra.upper()) - ord('A')
                contagem_letras[chr(65 + indice)][palavras.index(palavra)] += letra
    
    return contagem_letras

# Função para gerar o DataFrame e salvar no Excel
def gerar_excel(contagem_letras, nome_arquivo):
    # Criar o DataFrame
    df = pd.DataFrame(contagem_letras).T.fillna(0)
    
    # Salvar no Excel
    df.to_excel(nome_arquivo, index=False)

# Exemplo de uso
arquivo = '/home/roger/Documentos/python/hangmania/dicionario/br-sem-acentos.txt'  # Caminho para o arquivo de texto com palavras
nome_arquivo_excel = 'br-sem-acentos.xlsx'
contagem_letras = processar_palavras(arquivo)
gerar_excel(contagem_letras, nome_arquivo_excel)

print(f"Arquivo Excel '{nome_arquivo_excel}' criado.")
