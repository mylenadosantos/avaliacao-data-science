import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from statsmodels.stats.multicomp import pairwise_tukeyhsd

wine = load_wine(as_frame=True)
df = wine.frame

#Questão 4
#A:
amostra_estrat = df.groupby('target', group_keys=False).apply(
    lambda x: x.sample(frac=0.20, random_state=42)
)

n_amostra = len(amostra_estrat) 
print(f"Tamanho da população: {len(df)}")
print(f"Tamanho da amostra estratificada (20%): {n_amostra}")

#B:
plt.figure(figsize=(10, 5))
sns.histplot(df['proline'], kde=True, color='navy', label='População (N=178)', stat='density', alpha=0.4)
sns.histplot(amostra_estrat['proline'], kde=True, color='darkorange', label='Amostra 20% (n=36)', stat='density', alpha=0.5)

plt.title("Proline: População vs. Amostra Estratificada (20%)", fontsize=12, fontweight='bold')
plt.xlabel("Proline")
plt.ylabel("Densidade")
plt.legend()
plt.tight_layout()
plt.show()

#C:
rng = np.random.default_rng(42)

medias_amostrais = [
    rng.choice(amostra_estrat['proline'], size=n_amostra, replace=True).mean()
    for _ in range(1000)
]

sigma_pop = df['proline'].std(ddof=1)
std_obs = np.std(medias_amostrais, ddof=1)
std_teorico = sigma_pop / np.sqrt(n_amostra) 
fator_reducao = np.sqrt(n_amostra) 

print(f"Desvio-Padrão Populacional (σ): {sigma_pop:.2f}")
print(f"Desvio-Padrão Observado das 1.000 Médias: {std_obs:.2f}")
print(f"Desvio-Padrão Teórico (σ / √n): {std_teorico:.2f}")
print(f"Fator de redução (√n): {fator_reducao:.1f} vezes")

plt.figure(figsize=(10, 5))
sns.histplot(medias_amostrais, kde=True, color='teal', stat='density')
plt.axvline(df['proline'].mean(), color='red', linestyle='--', label=f'Média Populacional ({df["proline"].mean():.2f})')
plt.title("Distribuição das 1.000 Médias Amostrais de Proline (TCL, n=36)", fontsize=12, fontweight='bold')
plt.xlabel("Média da Variável Proline")
plt.ylabel("Densidade")
plt.legend()
plt.tight_layout()
plt.show()


#Questão 5

features = [col for col in df.columns if col != 'target']

#A:
stats_list = []

for var in features:
    grouped = df.groupby('target')[var]
    means = grouped.mean()
    medians = grouped.median()
    stds = grouped.std(ddof=1)
    cvs = (stds / means) * 100
    avg_cv = cvs.mean()
    
    stats_list.append({
        'variável': var,
        'cv_médio': avg_cv,
        'cv_cultivar_0': cvs.loc[0],
        'cv_cultivar_1': cvs.loc[1],
        'cv_cultivar_2': cvs.loc[2]
    })

df_stats = pd.DataFrame(stats_list).sort_values(by='cv_médio', ascending=False)

top3_vars = df_stats['variável'].head(3).tolist()

print("--- Top 3 Variáveis com Maior Coeficiente de Variação (CV) Médio ---")
print(df_stats[['variável', 'cv_médio']].head(3).to_string(index=False))

var_maior_cv = top3_vars[0]
print(f"\nA variável com o MAIOR CV médio é: '{var_maior_cv}' ({df_stats.iloc[0]['cv_médio']:.2f}%)\n")

#B:
outliers_contagem = {var: {} for var in top3_vars}

for var in top3_vars:
    for cult in [0, 1, 2]:
        sub = df[df['target'] == cult][var]
        q1 = sub.quantile(0.25)
        q3 = sub.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        outliers = sub[(sub < lower) | (sub > upper)]
        outliers_contagem[var][f'Cultivar {cult}'] = len(outliers)

df_outliers = pd.DataFrame(outliers_contagem)
print("--- Contagem de Outliers por Cultivar (Regra 1.5xIQR) ---")
print(df_outliers)

#C:
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.set_theme(style="whitegrid")

for i, var in enumerate(top3_vars):
    sns.boxplot(data=df, x='target', y=var, ax=axes[i], palette='Set2')
    axes[i].set_title(f"Boxplot de {var.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
    axes[i].set_xlabel("Cultivar")
    axes[i].set_ylabel(var)

plt.tight_layout()
plt.show()

#Questão 6

vars_q6 = ['magnesium', 'malic_acid', 'proanthocyanins', 'hue']

#A:
print("--- ITEM A: Assimetria ---")
for var in vars_q6:
    sk = df[var].skew()
    print(f"{var}: skewness = {sk:.4f}")

#B:
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')

titles = ['Magnesium', 'Malic Acid', 'Proanthocyanins', 'Hue']

for i, var in enumerate(vars_q6):
    ax = axes[i // 2, i % 2]
    sns.histplot(df[var], kde=True, ax=ax, color='teal', bins=15)
    ax.set_title(
        f'Histograma e Densidade (KDE) de {titles[i]}',
        fontsize=12,
        fontweight='bold',
    )
    ax.set_xlabel(var)
    ax.set_ylabel('Frequência')

plt.tight_layout()
plt.show()

#C:
print("\n--- ITEM C: Teste de Shapiro-Wilk ---")
for var in vars_q6:
    stat, p_val = stats.shapiro(df[var])
    print(f"{var}: W = {stat:.4f}, p-valor = {p_val:.6f}")

#Questão 7

features = wine.feature_names
n = len(df)

#A:
corr_matrix = df[features].corr(method='pearson')

print("--- ITEM A: Pares com |r| > 0.7 ---")
high_corr_pairs = []
for i in range(len(features)):
  for j in range(i + 1, len(features)):
    f1, f2 = features[i], features[j]
    r = corr_matrix.loc[f1, f2]
    if abs(r) > 0.7:
      high_corr_pairs.append((f1, f2, r))

for f1, f2, r in high_corr_pairs:
  is_excluded = set([f1, f2]) == {'flavanoids', 'total_phenols'}
  tag = " (EXCLUÍDO - analisado em aula)" if is_excluded else ""
  print(f"{f1} x {f2}: r = {r:.4f}{tag}")

#B:
var1, var2 = 'flavanoids', 'od280/od315_of_diluted_wines'
r_val = corr_matrix.loc[var1, var2]

df_degrees = n - 2
t_stat = r_val * np.sqrt(df_degrees) / np.sqrt(1 - r_val**2)
p_val_manual = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_degrees))

r_scipy, p_val_scipy = stats.pearsonr(df[var1], df[var2])

print("\n--- ITEM B: Teste de Significância ---")
print(f"Estatística t (manual): {t_stat:.4f}")
print(f"p-valor (manual): {p_val_manual:.4e}")
print(f"scipy pearsonr: r = {r_scipy:.4f}, p-valor = {p_val_scipy:.4e}")

#C:
w1, p1 = stats.shapiro(df[var1])
w2, p2 = stats.shapiro(df[var2])

rho, p_spearman = stats.spearmanr(df[var1], df[var2])
tau, p_kendall = stats.kendalltau(df[var1], df[var2])

print("\n--- ITEM C: Shapiro-Wilk, Spearman e Kendall ---")
print(f"Shapiro {var1}: W = {w1:.4f}, p-valor = {p1:.4e}")
print(f"Shapiro {var2}: W = {w2:.4f}, p-valor = {p2:.4e}")
print(f"Spearman: rho = {rho:.4f}, p-valor = {p_spearman:.4e}")
print(f"Kendall: tau = {tau:.4f}, p-valor = {p_kendall:.4e}")

#D:
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    ax=axes[0],
    cbar_kws={'label': 'r de Pearson'},
)
axes[0].set_title(
    'Heatmap da Matriz de Correlação Completa', fontsize=14, fontweight='bold'
)

sns.scatterplot(
    data=df,
    x=var1,
    y=var2,
    hue='target',
    palette='Set2',
    ax=axes[1],
    s=70,
    alpha=0.8,
)
sns.regplot(
    data=df,
    x=var1,
    y=var2,
    scatter=False,
    ax=axes[1],
    color='darkblue',
    line_kws={
        'linewidth': 2,
        'linestyle': '--',
        'label': 'Reta de Regressão Global',
    },
)

axes[1].set_title(
    f'Scatter Plot: {var1} vs {var2}', fontsize=14, fontweight='bold'
)
axes[1].set_xlabel(var1, fontsize=12)
axes[1].set_ylabel(var2, fontsize=12)
axes[1].legend(title='Cultivar', frameon=True)

plt.tight_layout()
plt.show()

#Questão 8:

cultivar_b = df[df['target'] == 1]['magnesium']
cultivar_c = df[df['target'] == 2]['magnesium']

#A:
print('--- Cultivar B ---')
print(
    f'Média: {cultivar_b.mean():.2f}, Desvio Padrão: {cultivar_b.std():.2f},'
    f' Mediana: {cultivar_b.median():.2f}'
)

print('\n--- Cultivar C ---')
print(
    f'Média: {cultivar_c.mean():.2f}, Desvio Padrão: {cultivar_c.std():.2f},'
    f' Mediana: {cultivar_c.median():.2f}'
)

#B:
w_b, p_sw_b = stats.shapiro(cultivar_b)
w_c, p_sw_c = stats.shapiro(cultivar_c)
stat_lev, p_lev = stats.levene(cultivar_b, cultivar_c)

print('\n--- Teste de Normalidade (Shapiro-Wilk) ---')
print(f'Cultivar B: W = {w_b:.4f}, p-valor = {p_sw_b:.4e}')
print(f'Cultivar C: W = {w_c:.4f}, p-valor = {p_sw_c:.4e}')

print('\n--- Homogeneidade de Variâncias (Levene) ---')
print(f'Estatística = {stat_lev:.4f}, p-valor = {p_lev:.4f}')

#C:
normal_b = p_sw_b > 0.05
normal_c = p_sw_c > 0.05
variancias_iguais = p_lev > 0.05

print('\n--- Decisão do teste ---')
if normal_b and normal_c and variancias_iguais:
    print('Pressupostos atendidos → Teste t de Student (equal_var=True)')
    stat, p = stats.ttest_ind(cultivar_b, cultivar_c, equal_var=True)
    print(f't = {stat:.4f}, p-valor = {p:.4e}')
elif normal_b and normal_c and not variancias_iguais:
    print('Normalidade OK, variâncias diferentes → Teste t de Welch (equal_var=False)')
    stat, p = stats.ttest_ind(cultivar_b, cultivar_c, equal_var=False)
    print(f't = {stat:.4f}, p-valor = {p:.4e}')
else:
    print('Normalidade violada → Mann-Whitney U (não paramétrico)')
    stat, p = stats.mannwhitneyu(cultivar_b, cultivar_c, alternative='two-sided')
    print(f'U = {stat:.4f}, p-valor = {p:.4e}')

#D:
df_bc = df[df['target'].isin([1, 2])].copy()
df_bc['Cultivar'] = df_bc['target'].map({1: 'Cultivar B', 2: 'Cultivar C'})

plt.figure(figsize=(7, 6))
sns.boxplot(x='Cultivar', y='magnesium', data=df_bc)
sns.stripplot(
    x='Cultivar',
    y='magnesium',
    data=df_bc,
    color='black',
    alpha=0.5,
    jitter=0.2,
    size=6,
)

plt.title(
    'Comparação do Teor de Magnésio: Cultivar B vs Cultivar C',
    fontsize=13,
    fontweight='bold',
)
plt.xlabel('Cultivar', fontsize=11)
plt.ylabel('Teor de Magnésio (magnesium)', fontsize=11)
plt.tight_layout()
plt.show()

#Questão 9

df['cultivar_label'] = df['target'].map(
    {0: 'Cultivar A', 1: 'Cultivar B', 2: 'Cultivar C'}
)

g0 = df[df['target'] == 0]['malic_acid']
g1 = df[df['target'] == 1]['malic_acid']
g2 = df[df['target'] == 2]['malic_acid']

#A:
print('--- ITEM A: Normalidade (Shapiro-Wilk) ---')
print('Cultivar A:', stats.shapiro(g0))
print('Cultivar B:', stats.shapiro(g1))
print('Cultivar C:', stats.shapiro(g2))

print('\n--- ITEM A: Homocedasticidade (Levene) ---')
print('Levene:', stats.levene(g0, g1, g2))

#B:
print('\n--- ITEM B: Kruskal-Wallis ---')
print('Kruskal-Wallis:', stats.kruskal(g0, g1, g2))

#C:
print('\n--- ITEM C: Post-Hoc Tukey HSD ---')
tukey = pairwise_tukeyhsd(
    endog=df['malic_acid'], groups=df['cultivar_label'], alpha=0.05
)
print(tukey)

#D:
plt.figure(figsize=(8, 6))
sns.boxplot(x='cultivar_label', y='malic_acid', data=df,
            order=['Cultivar A', 'Cultivar B', 'Cultivar C'],
            medianprops=dict(color='red', linewidth=2))
sns.stripplot(x='cultivar_label', y='malic_acid', data=df,
              order=['Cultivar A', 'Cultivar B', 'Cultivar C'],
              color='black', alpha=0.4, jitter=0.2, size=4)
plt.title('Comparação do Teor de Ácido Málico', fontsize=13, fontweight='bold')
plt.xlabel('Cultivar')
plt.ylabel('Teor de Ácido Málico (malic_acid)')
plt.tight_layout()
plt.show()

#Questão 10

df['cultivar_label'] = df['target'].map(
    {0: 'Cultivar A', 1: 'Cultivar B', 2: 'Cultivar C'}
)

#A:
amostra, _ = train_test_split(
    df, train_size=0.6, stratify=df['target'], random_state=7
)

print('--- ITEM A: Amostra Estratificada ---')
print('Tamanho da amostra:', len(amostra))
print(amostra['cultivar_label'].value_counts())

#B:
num_cols = wine.feature_names
corr_matrix = amostra[num_cols].corr().abs()

for col in num_cols:
    corr_matrix.loc[col, col] = 0

excluidos = [
    {'flavanoids', 'total_phenols'},
    {'flavanoids', 'od280/od315_of_diluted_wines'},
    {'total_phenols', 'od280/od315_of_diluted_wines'},
]

for v1, v2 in excluidos:
    corr_matrix.loc[v1, v2] = 0
    corr_matrix.loc[v2, v1] = 0

par_max = corr_matrix.unstack().idxmax()
val_max = corr_matrix.unstack().max()

print('\n--- ITEM B: Maior Redundância (fora Q7) ---')
print(f'Par: {par_max} | Correlação: {val_max:.4f}')


#C
g0_alc = amostra[amostra['target'] == 0]['alcohol']
g1_alc = amostra[amostra['target'] == 1]['alcohol']
g2_alc = amostra[amostra['target'] == 2]['alcohol']

print('\n--- ITEM C: Pressupostos ---')
print('Shapiro Cultivar A:', stats.shapiro(g0_alc))
print('Shapiro Cultivar B:', stats.shapiro(g1_alc))
print('Shapiro Cultivar C:', stats.shapiro(g2_alc))
print('Levene:', stats.levene(g0_alc, g1_alc, g2_alc))

f_stat, p_anova = stats.f_oneway(g0_alc, g1_alc, g2_alc)
print(f'\nANOVA: F = {f_stat:.4f}, p-valor = {p_anova:.4e}')

tukey = pairwise_tukeyhsd(
    endog=amostra['alcohol'], groups=amostra['cultivar_label'], alpha=0.05
)
print('\n--- Post-Hoc Tukey HSD ---')
print(tukey)

#D
plt.figure(figsize=(8, 6))
sns.boxplot(
    x='cultivar_label', y='alcohol', data=amostra,
    order=['Cultivar A', 'Cultivar B', 'Cultivar C'],
    medianprops=dict(color='red', linewidth=2)
)
sns.stripplot(
    x='cultivar_label', y='alcohol', data=amostra,
    order=['Cultivar A', 'Cultivar B', 'Cultivar C'],
    color='black', alpha=0.4, jitter=0.2, size=4
)
plt.title('Teor Alcoólico por Cultivar (Amostra 60%)', fontsize=13, fontweight='bold')
plt.xlabel('Cultivar')
plt.ylabel('Teor Alcoólico (% vol.)')
plt.tight_layout()
plt.show()