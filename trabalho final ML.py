# %% [markdown]
# # Trabalho Final - Machine Learning
# 
# * Arthur Bernardes Suematsu
# * Eduardo Henrique Silva Souza
# * Gustavo Almeida Brezzi
# * Wagner Sanches Gonçalves

# %% [markdown]
# Neste trabalho, como parte do time de analistas da Quantum Finance, vocês deverão explorar uma base de dados originalmente utilizada para classificação de score de crédito, disponível no Kaggle (https://www.kaggle.com/datasets/parisrohan/credit-score-classification), utilizando técnicas de Análise Exploratória de Dados (EDA) e algoritmos de Machine Learning supervisionados. 
# 
# 
# O objetivo é aplicar e interpretar os resultados obtidos, assim como criar um sistema que gere valor a partir da análise da base de dados.
# 
# **Modelo de Classificação Supervisionada**
# 
# 
# Desenvolver um modelo de classificação supervisionada para prever a classificação de crédito dos indivíduos presentes na base.
# 
# 
# Passos esperados:
# 
# 1. Realizar uma análise exploratória dos dados (EDA) para entender as características principais da base e as relações entre variáveis; 2 pontos
# 
# 2. Implementar um pipeline de modelo de classificação usando Random Forest, XGBoost e LightGBM. Use GridSearch para otimizar os parametros de cada modelo; 4 pontos
# 
# 3. Avaliar os resultados utilizando a métrica mais adequada e **justifique** sua escolha; 2 pontos
# 
# 4. Apresentar os resultados, indicando a métrica no conjunto de treino (train.csv) e explicar como o modelo pode ser utilizado para decisões financeiras. 2 pontos
# 
# 
# Entregáveis:
# 
# 1. Este jupyter notebook executado e com os resultados aparentes (não serão aceitoa outros formatos)

# %% [markdown]
# ### Importação das bibliotecas necessárias

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### Carregamento e exploração inicial dos dados

# %%
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Verificando as dimensões dos dados
print(f"Dimensões do conjunto de treino: {train_data.shape}")
print(f"Dimensões do conjunto de teste: {test_data.shape}")

# %%
train_data.head()

# %%
train_data.columns.tolist()


# %%
test_data.head()

# %%
test_data.columns.tolist()

# %%
# Observando a diferença entre os conjuntos (teste não tem Credit_Score)
train_cols = set(train_data.columns.tolist())
test_cols = set(test_data.columns.tolist())
print("Colunas apenas no treino:", train_cols - test_cols)
print("Colunas apenas no teste:", test_cols - train_cols)

# %% [markdown]
# ### Análise exploratória de dados (EDA)

# %%
train_data.info()

# %%
train_data.describe()

# %%
train_data.isnull().sum()

# %%
train_data['Credit_Score'].unique()

# %%
# Verificando a distribuição da variável alvo
print("\nDistribuição da variável alvo (Credit_Score):")
print(train_data['Credit_Score'].value_counts())
print(train_data['Credit_Score'].value_counts(normalize=True) * 100)

# %% [markdown]
# Pré-processamento

# %%
def duplicate_values(df):
    print("Validação de duplicação.")
    num_duplicates = df.duplicated(subset=None, keep='first').sum()
    if num_duplicates > 0:
        print("Existem", num_duplicates, "duplicadas.")
        df.drop_duplicates(keep='first', inplace=True)
        print(num_duplicates, "duplicadas excluidas")
    else:
        print("Não existem duplicadas")

duplicate_values(train_data)
duplicate_values(test_data)

# %%
# Verificando valores inválidos ou inconsistentes em colunas numéricas
print("Verificando valores numéricos inválidos...")
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    invalid_values = train_data[~train_data[col].astype(str).str.replace('.', '').str.replace('-', '').str.isdigit()]
    if not invalid_values.empty:
        print(f"Coluna {col} contém valores não numéricos: {invalid_values[col].unique()}")

# %%
train_data['Age'].unique()

# %%
# Função para limpar e converter valores
def clean_data(df):
    df_copy = df.copy()

    # colunas que não serão usadas
    df_copy = df_copy.drop(['ID','Customer_ID','Month','Name','SSN', 'Type_of_Loan', 'Changed_Credit_Limit', 'Monthly_Inhand_Salary'], axis = 1)

    # preenche o na do Credit_Mix com desconhecido
    df_copy['Credit_Mix'].fillna('Unknown', inplace=True)
    df_copy['Credit_Mix'].astype('object')

    df_copy.loc[pd.isna(df_copy['Occupation']), 'Occupation'] = 'Other'
   
    # Limpando e convertendo a coluna Age para numérico
    df_copy['Age'] = pd.to_numeric(df_copy['Age'].astype(str).str.replace('_', ''), errors='coerce')
    # Substituir valores inválidos (negativos ou muito altos) por NaN
    df_copy.loc[df_copy['Age'] < 0, 'Age'] = np.nan
    df_copy.loc[df_copy['Age'] > 100, 'Age'] = np.nan

    # Preencher NaN com a mediana das idades válidas
    mediana_idade = df_copy['Age'].median()
    df_copy['Age'].fillna(mediana_idade, inplace=True)
    
    # Convertendo Credit_History_Age para numérico (em meses)
    def convert_credit_history(x):
        if pd.isna(x) or x == 'NA':
            return np.nan
        try:
            years = 0
            months = 0
            if 'Years' in str(x):
                years = int(str(x).split('Years')[0].strip())
            if 'Months' in str(x):
                months = int(str(x).split('Months')[0].split('and')[-1].strip())
            return years * 12 + months
        except:
            return np.nan
    
    df_copy['Credit_History_Age'] = df_copy['Credit_History_Age'].apply(convert_credit_history)
    
    # Tratando valores especiais como NaN
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].replace('_', np.nan)
        df_copy[col] = df_copy[col].replace('!@9#%8', np.nan)
        df_copy[col] = df_copy[col].replace('#F%$D@*&8', np.nan)
    
    # Convertendo colunas numéricas - primeiro limpando caracteres especiais
    numeric_cols = ['Age','Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Credit_History_Age',
                  'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
                  'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                  'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                  'Amount_invested_monthly', 'Monthly_Balance']
    
    for col in numeric_cols:
        # Primeiro limpar quaisquer caracteres não numéricos
        if col in df_copy.columns:
            # Converter para string, remover caracteres não numéricos exceto ponto decimal
            df_copy[col] = df_copy[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True)
            # Converter para float
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    return df_copy

# %%
# Aplicando a limpeza aos conjuntos de dados
train_data = clean_data(train_data)
test_data = clean_data(test_data)

# %%
test_data.info()

# %%
train_data.info()

# %%
# Usando dicionário para mapear categorias para valores numéricos
mapeamento = {'Good': 2, 'Standard': 1, 'Poor': 0}

# Aplicando o mapeamento 
train_data['Credit_Score'] = train_data['Credit_Score'].map(mapeamento)

# %%
# Verificando valores nulos após limpeza
print("\nValores nulos após limpeza (treino):")
print(train_data.isnull().sum())
print("\nValores nulos após limpeza (teste):")
print(test_data.isnull().sum())

# %%
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Credit_Score', data=train_data)
plt.title('Distribuição da Classificação de Crédito')
plt.ylabel('Quantidade')
plt.xlabel('Classificação de Crédito (0=Poor, 1=Standard, 2=Good)')
plt.show()

# %%
# Verificando a distribuição da idade
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'], kde=True)
plt.title('Distribuição de Idade')
plt.plot()

# %%
# rows_with_empty_values = train_data[train_data.eq('').any(axis=1)]

# train_data.loc[rows_with_empty_values.index] = train_data.loc[rows_with_empty_values.index].replace('', np.nan)


# %%


# %%
train_data['Credit_Mix'].unique()

# %% [markdown]
# #### Quanto maior o histórico de crédito melhor a classificação

# %%
# Explorando relação entre histórico de crédito e score
plt.figure(figsize=(12, 6))
sns.boxplot(x='Credit_Score', y='Credit_History_Age', data=train_data)
plt.title('Histórico de Crédito por Classificação de Crédito')
plt.plot()

# %%
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
corr_matrix = train_data[numeric_cols].corr()

# Visualizando as correlações 
correlacoes_com_target = corr_matrix['Credit_Score'].sort_values(ascending=False)
print(correlacoes_com_target)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.savefig('matriz_correlacao.png')
plt.show()

# %%
print(train_data['Delay_from_due_date'].unique())

# %%
# Explorando a relação entre atraso de pagamento e pontuação de crédito
plt.figure(figsize=(12, 6))
sns.boxplot(x='Credit_Score', y='Delay_from_due_date', data=train_data)
plt.title('Atraso de Pagamento por Classificação de Crédito')
plt.xlabel('Classificação de Crédito')
plt.ylabel('Atraso de Pagamento (dias)')
plt.show()


# %% [markdown]
# Isso mostra que o atraso no pagamento auxlia na decisão de aprovar ou não o crédito

# %%
print(train_data['Outstanding_Debt'].unique())

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='Credit_Score', y='Outstanding_Debt', data=train_data)
plt.title('Dívida Pendentes por Classificação de Crédito')
plt.xlabel('Classificação de Crédito')
plt.ylabel('Dívida Pendentes')
plt.show()


# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='Credit_Score', y='Age', data=train_data)
plt.title('Idade por Classificação de Crédito')
plt.xlabel('Classificação de Crédito')
plt.ylabel('Idade')
plt.show()


# %%
colunas_obj = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
for coluna in colunas_obj:
    print(f"Valores únicos na coluna {coluna}: {train_data[coluna].unique()}")


# %%


# Criar um dicionário para armazenar os encoders
encoders = {}

# Colunas categóricas a serem codificadas
colunas_categoricas = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

# Aplicar Label Encoding em cada coluna categórica
for coluna in colunas_categoricas:
    le = LabelEncoder()
    # Preencher valores NaN com uma string para codificação
    train_data[coluna] = train_data[coluna].fillna('Missing')
    train_data[coluna] = le.fit_transform(train_data[coluna])
    encoders[coluna] = le

# Exibir os valores codificados para verificação
for coluna in colunas_categoricas:
    print(f"Valores codificados na coluna {coluna}: {train_data[coluna].unique()}")


# %%
# Calcular a correlação entre as colunas codificadas e a coluna 'Credit_Score'
correlacoes = {}
for coluna in colunas_categoricas:
    correlacao = train_data[coluna].corr(train_data['Credit_Score'])
    correlacoes[coluna] = correlacao

# Exibir a lista de correlações
for coluna, correlacao in correlacoes.items():
    print(f"Correlação entre {coluna} e Credit_Score: {correlacao}")


# %%
# Calcular a correlação entre todas as colunas e a coluna 'Credit_Score'
correlacoes_todas = train_data.corr()['Credit_Score']
correlacoes_todas.sort_values(ascending=False)

# %%
print("\nVerificando valores numéricos inválidos...")
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    invalid_values = train_data[~train_data[col].astype(str).str.replace('.', '').str.replace('-', '').str.isdigit()]
    if not invalid_values.empty:
        print(f"Coluna {col} contém valores não numéricos: {invalid_values[col].unique()}")

# %%
print(train_data.isnull().sum())

# %%
# Separando features e target
X = train_data.drop(['Credit_Score'], axis=1)
y = train_data['Credit_Score']

# %%
# Visualizando dados após a limpeza
print("\nFormato dos dados após limpeza:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

print("\nTipos de dados após limpeza:")
print(X.dtypes)

# %%
# Definição das features
features = [col for col in train_data.columns if col != 'Credit_Score']

# Preprocessador para features numéricas
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, features)])


X_transformado = preprocessor.fit_transform(X)
# Usar weighted average (considera a proporção de cada classe)


# %%
# Verificar a distribuição das classes
print("Distribuição das classes:")
print(y.value_counts())
print(f"Proporção da classe 0 (poor): {(y == 0).mean():.4f}")

# Verificar se há valores faltantes
print(f"\nValores faltantes em y: {y.isna().sum()}")
print(f"Valores faltantes em X_transformado: {np.isnan(X_transformado).sum()}")



# %%
f1_scorer = make_scorer(f1_score, average='weighted')

# %%


# Pipeline com Random Forest
pipeline_rf = Pipeline([
    ('classificador', RandomForestClassifier(random_state=42))
])

# Parâmetros para execução
parametros_rf = {
  'classificador__n_estimators': [100, 200, 300],
    'classificador__max_depth': [None, 10, 20],
    'classificador__min_samples_split': [2, 5, 10]
}

# GridSearch 
grid_rf = GridSearchCV(pipeline_rf, parametros_rf, cv=3, scoring=f1_scorer, n_jobs=-1)

# Treinamento
print("Treinando Random Forest...")
grid_rf.fit(X_transformado, y)

# Resultados
print("\n--- Resultados Random Forest ---")
print(f"Melhor f1-score: {grid_rf.best_score_:.4f}")
print(f"Melhores parâmetros: {grid_rf.best_params_}")

# %%
# Pipeline com XGBoost
pipeline_xgb = Pipeline([
    ('classificador', xgb.XGBClassifier(random_state=42))
])

# Parâmetros para execução
parametros_xgb = {
    'classificador__n_estimators': [100, 200],
    'classificador__max_depth': [3, 5, 7],
    'classificador__learning_rate': [0.01, 0.1]
}

# GridSearch
grid_xgb = GridSearchCV(pipeline_xgb, parametros_xgb, cv=3, scoring=f1_scorer, n_jobs=-1)

# Treinamento
print("Treinando XGBoost...")
grid_xgb.fit(X_transformado, y)

# Resultados
print("\n--- Resultados XGBoost ---")
print(f"Melhor f1-score: {grid_xgb.best_score_:.4f}")
print(f"Melhores parâmetros: {grid_xgb.best_params_}")

# %%
# Pipeline com LightGBM
pipeline_lgbm = Pipeline([
    ('classificador', lgb.LGBMClassifier(random_state=42))
])

# Parâmetros para execução
parametros_lgbm = {
    'classificador__n_estimators': [100, 200],
    'classificador__max_depth': [3, 5, 7],
    'classificador__learning_rate': [0.01, 0.1]
}

# GridSearch
grid_lgbm = GridSearchCV(pipeline_lgbm, parametros_lgbm, cv=3, scoring=f1_scorer, n_jobs=-1)

# Treinamento
print("Treinando LightGBM...")
grid_lgbm.fit(X_transformado, y)

# Resultados
print("\n--- Resultados LightGBM ---")
print(f"Melhor f1-score: {grid_lgbm.best_score_:.4f}")
print(f"Melhores parâmetros: {grid_lgbm.best_params_}")

# %%
# Determinação do melhor modelo
modelos = {'Random Forest': grid_rf, 'XGBoost': grid_xgb, 'LightGBM': grid_lgbm}
melhor_modelo = max(modelos.items(), key=lambda x: x[1].best_score_)
print(f"\nMelhor modelo: {melhor_modelo[0]} com f1-score de {melhor_modelo[1].best_score_:.4f}")

# %%
# Função para avaliar modelo com múltiplas métricas
def avaliar_modelo(modelo, X, y, nome_modelo):
    # Fazer predições
    y_pred = modelo.predict(X)
    y_proba = modelo.predict_proba(X)
    
    # Calcular métricas
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    # Para ROC AUC em multiclasse, usamos One-vs-Rest
    try:
        roc_auc = roc_auc_score(y, y_proba, multi_class='ovr')
    except:
        roc_auc = "N/A"
    
    # Matriz de confusão
    cm = confusion_matrix(y, y_pred)
    
    # Exibir resultados
    print(f"\n--- Resultados {nome_modelo} ---")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"ROC AUC (OvR): {roc_auc}")
    
    # Exibir matriz de confusão
    print("\nMatriz de Confusão:")
    print(cm)
    
    return {
        'modelo': nome_modelo,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# %%
# Avaliar cada modelo após treinar
resultados_rf = avaliar_modelo(grid_rf.best_estimator_, X_transformado, y, "Random Forest")

# %%
resultados_xgb = avaliar_modelo(grid_xgb.best_estimator_, X_transformado, y, "XGBoost")

# %%
resultados_lgbm = avaliar_modelo(grid_lgbm.best_estimator_, X_transformado, y, "LightGBM")

# %%

# DataFrame para comparar todos os modelos
resultados_df = pd.DataFrame([resultados_rf, resultados_xgb, resultados_lgbm])

resultados_df

# %%
# Visualizar comparação das métricas
metricas = ['accuracy', 'precision', 'recall', 'f1']
resultados_plot = resultados_df.set_index('modelo')[metricas]

# %%
plt.figure(figsize=(12, 6))
resultados_plot.plot(kind='bar')
plt.title('Comparação de Métricas por Modelo')
plt.ylabel('Valor')
plt.xlabel('Modelo')
plt.ylim(0, 1)
plt.legend(title='Métrica')
plt.tight_layout()
plt.show()

# %%


# Dicionário com seus grids já treinados
models = {
    'RandomForest': grid_rf,
    'XGBoost':      grid_xgb,
    'LightGBM':     grid_lgbm
}

# Nomes na ordem dos rótulos 0,1,2
label_names = ['Poor', 'Standard', 'Good']

f1_scores_poor = {}
for name, grid in models.items():
    # Previsões sobre X_transformado
    y_pred = grid.predict(X_transformado)
    report = classification_report(
        y,
        y_pred,
        labels=[0, 1, 2],           # especifica que as classes são 0,1,2
        target_names=label_names,   # mapeia 0→'Poor', 1→'Standard', 2→'Good'
        output_dict=True
    )
    f1_scores_poor[name] = report['Poor']['f1-score']

# Monta um DataFrame pra visualização
df_scores = pd.Series(f1_scores_poor, name='F1-score (Poor)').to_frame()
print(df_scores)


# %% [markdown]
# ## 3. Avaliação e seleção do modelo final
# 
# **Métrica principal:** F1-score da classe “Poor”  
# _Justificativa:_ como o custo de conceder crédito a um cliente de alto risco (falso positivo) é muito superior ao de negar crédito a bom pagador (falso negativo), priorizamos o F1-score na classe “Poor”, que combina precision (evita falsos positivos) e recall (captura a maior parte dos riscos).
# 
# | Modelo       | F1-score (Poor) |
# |--------------|-----------------|
# | RandomForest | **0.755235**      |
# | LightGBM     | 0.718871          |
# | XGBoost      | 0.737862          |
# 
# > **Modelo escolhido:** RandomForest, por apresentar o maior F1-score (Poor).

# %% [markdown]
# ## 4. Aplicação financeira do modelo
# 
# O RandomForest treinado, com F1-score de 0.75 na classe “Poor”, pode ser integrado ao fluxo de concessão de crédito da Quantum Finance da seguinte forma:
# 
# - **Score de risco:** ao receber a solicitação de crédito, o sistema calcula P(“Poor”).  
# - **Negação automática:** se P(“Poor”) ≥ 0.5, o crédito é automaticamente negado ou passa por análise manual.  
# - **Limites e juros dinâmicos:** para 0.2 ≤ P(“Poor”) < 0.5, concede-se crédito com limites e taxas ajustados ao perfil de risco.  
# - **Monitoramento contínuo:** clientes com P(“Poor”) crescente são acionados por cartas de revisão de condições ou monitorados por squads de cobrança.
# 
# Dessa forma, o modelo gera valor ao evitar perdas financeiras (minimizando falsos positivos) e ao maximizar oportunidades de negócios para clientes de baixo risco.
# 

# %% [markdown]
# 


