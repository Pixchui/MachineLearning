import pandas as pd
import sklearn as sk

df = pd.read_csv('dados_jogos_prontos.csv')

#Identificar valores 0 e 1 nas colunas de gênero e plataforma
#Remover as linhas com valor 0 nas colunas

for coluna in df.columns:
    if coluna.startswith('Gênero_') or coluna.startswith('Plataforma_'):
        df = df[df[coluna] != 0]