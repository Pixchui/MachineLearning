import pandas as pd
import sklearn as sk

df = pd.read_csv('dados_jogos_prontos.csv')

#Print nas linhas duplicadas

linhas_duplicadas = df[df.duplicated()]

print(linhas_duplicadas)

#Removendo as linhas duplicadas

df_sem_duplicatas = df.drop_duplicates()

print(df_sem_duplicatas.duplicated().sum())
