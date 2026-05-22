import pandas as pd
import sklearn as sk

df = pd.read_csv('game_dataset.csv')

media_notas = df['Classificação de Usuários'].mean()
print(f'Média das notas dos usuários: {media_notas}')

df['Classificação de Usuários'] = df['Classificação de Usuários'].fillna(media_notas)

mediana_idade = df['Idade Recomendada'].median()
print(f'Mediana da idade recomendada: {mediana_idade}')

df['Idade Recomendada'] = df['Idade Recomendada'].fillna(mediana_idade)

print(df.isnull().sum())

df['Lançamento no Brasil'] = df['Lançamento no Brasil'].map({'Sim': 1, 'Não': 0})

print(df['Lançamento no Brasil'].value_counts())

df_transformado = pd.get_dummies(df, columns=['Gênero', 'Plataforma'])



for coluna in df_transformado.columns:
    if coluna.startswith('Gênero_') or coluna.startswith('Plataforma_'):
        df_transformado[coluna] = df_transformado[coluna].map({True: 1, False: 0})

from sklearn.preprocessing import MinMaxScaler

normalizador = MinMaxScaler()

colunas_para_normalizar = ['Vendas Globais', 'Preço']
df_transformado[colunas_para_normalizar] = normalizador.fit_transform(
    df_transformado[colunas_para_normalizar]
)

print(df_transformado[colunas_para_normalizar].describe())

df_transformado.to_csv('dados_jogos_prontos.csv', index=False)