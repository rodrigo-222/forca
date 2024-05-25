import pandas as pd

def contar_letras(arquivo):
    # Ler o arquivo linha por linha
    with open(arquivo, 'r') as file:
        palavras = file.read().splitlines()
        lista = []  # Lista para armazenar todas as informações
        for i,palavra in enumerate(palavras):
            print(i)
            palavra = palavra.lower()  # Converter para minúsculas
            contagens = [0]*26  # Inicializar contadores para cada letra
            for letra in palavra:
                if 'a' <= letra <= 'z':  # Verificar se a letra é alfabética
                    indice = ord(letra) - ord('a')
                    contagens[indice] += 1
            # Adicionar a palavra e suas contagens como uma sublista na lista principal
            lista.append([palavra] + contagens)

        # Criar DataFrame a partir da lista
            df = pd.DataFrame(lista, columns=['Palavra'] + list(map(chr, range(ord('a'), ord('z')+1))))
        
        # Salvar o DataFrame em um arquivo Excel
            
        salvar_excel(df, i)

def salvar_excel(df,i):
    df.to_excel(f"resultado{i}.xlsx", index=False)

# Caminho do arquivo
arquivo = '/home/roger/Documentos/python/hangmania/dicionario/br-sem-acentos.txt'
contar_letras(arquivo)
