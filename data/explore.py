import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/training_data.csv")

# 1. Visão geral
print(df.shape)              # esperado: ~942 linhas (um doente por linha)
print(df.columns.tolist())
print(df.head())
df.info()
print(df.isna().sum())       # valores em falta por coluna

# 2. Garantir uma linha por doente
print(df["Patient ID"].is_unique)

# 3. Distribuição da classe alvo (Murmur)
print(df["Murmur"].value_counts())
print(df["Murmur"].value_counts(normalize=True).round(3))
df["Murmur"].value_counts().plot(kind="bar", title="Murmur")
plt.show()

# 4. Gravações por doente (a coluna lista os locais, ex.: "AV+PV+TV+MV")
df["n_rec_locais"] = df["Recording locations:"].str.split("+").str.len()
print(df["n_rec_locais"].value_counts().sort_index())
print(pd.crosstab(df["n_rec_locais"], df["Murmur"]))

# 5. Locais onde há sopro (só faz sentido para Present)
print(df.loc[df["Murmur"] == "Present", "Murmur locations"].value_counts().head(10))

# 6. Demografia vs. classe
print(df.groupby("Murmur")["Age"].value_counts().unstack(0))   # grupos etários
print(pd.crosstab(df["Sex"], df["Murmur"], normalize="index").round(3))
print(df.groupby("Murmur")[["Height", "Weight"]].describe().T)

# 7. Campanha (as duas campanhas podem ter distribuições diferentes)
print(pd.crosstab(df["Campaign"], df["Murmur"]))