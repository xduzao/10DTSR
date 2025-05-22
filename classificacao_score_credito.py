#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
# Seção 1: Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as XGBClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# %%
# Seção 2: Carregamento e exploração inicial dos dados
print("Carregando os dados...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Verificando as dimensões dos dados
print(f"Dimensões do conjunto de treino: {train_data.shape}")
print(f"Dimensões do conjunto de teste: {test_data.shape}")

# Verificando as primeiras linhas do conjunto de treino
print("\nPrimeiras linhas do conjunto de treino:")
print(train_data.head())

# Verificando as colunas do conjunto de treino
print("\nColunas do conjunto de treino:")
print(train_data.columns.tolist())

# Verificando as colunas do conjunto de teste
print("\nColunas do conjunto de teste:")
print(test_data.columns.tolist())

# Observando a diferença entre os conjuntos (teste não tem Credit_Score)
print("\nDiferença entre as colunas dos conjuntos:")
train_cols = set(train_data.columns.tolist())
test_cols = set(test_data.columns.tolist())
print("Colunas apenas no treino:", train_cols - test_cols)
print("Colunas apenas no teste:", test_cols - train_cols)

# %%
# Seção 3: Análise exploratória de dados (EDA)
print("\nRealizando análise exploratória dos dados...")

# Verificando informações gerais dos dados
print("\nInformações gerais do conjunto de treino:")
print(train_data.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas do conjunto de treino:")
print(train_data.describe())

# Verificando valores ausentes
print("\nQuantidade de valores ausentes por coluna:")
print(train_data.isnull().sum())

# Verificando a distribuição da variável alvo
print("\nDistribuição da variável alvo (Credit_Score):")
print(train_data['Credit_Score'].value_counts())
print(train_data['Credit_Score'].value_counts(normalize=True) * 100)

# Visualizando a distribuição da variável alvo
plt.figure(figsize=(10, 6))
sns.countplot(x='Credit_Score', data=train_data)
plt.title('Distribuição da Classificação de Crédito')
plt.ylabel('Quantidade')
plt.savefig('distribuicao_credit_score.png')
plt.close()

# Verificando valores inválidos ou inconsistentes em colunas numéricas
print("\nVerificando valores numéricos inválidos...")
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    invalid_values = train_data[~train_data[col].astype(str).str.replace('.', '').str.replace('-', '').str.isdigit()]
    if not invalid_values.empty:
        print(f"Coluna {col} contém valores não numéricos: {invalid_values[col].unique()}")

# Verificando a distribuição da idade
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'], kde=True)
plt.title('Distribuição de Idade')
plt.savefig('distribuicao_idade.png')
plt.close()

# Explorando relação entre renda anual e score de crédito
plt.figure(figsize=(12, 6))
sns.boxplot(x='Credit_Score', y='Annual_Income', data=train_data)
plt.title('Renda Anual por Classificação de Crédito')
plt.savefig('renda_por_score.png')
plt.close()

# Explorando relação entre histórico de crédito e score
plt.figure(figsize=(12, 6))
sns.boxplot(x='Credit_Score', y='Credit_History_Age', data=train_data)
plt.title('Histórico de Crédito por Classificação de Crédito')
plt.savefig('historico_por_score.png')
plt.close()

# Matriz de correlação para variáveis numéricas
plt.figure(figsize=(16, 12))
corr_matrix = train_data.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.savefig('matriz_correlacao.png')
plt.close()

# %%
# Seção 4: Pré-processamento dos dados
print("\nRealizando pré-processamento dos dados...")

# Função para limpar e converter valores
def clean_data(df):
    df_copy = df.copy()
    
    # Limpando e convertendo a coluna Age para numérico
    df_copy['Age'] = pd.to_numeric(df_copy['Age'].astype(str).str.replace('_', ''), errors='coerce')
    
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
    
    df_copy['Credit_History_Age_Months'] = df_copy['Credit_History_Age'].apply(convert_credit_history)
    
    # Tratando valores especiais como NaN
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].replace('_', np.nan)
        df_copy[col] = df_copy[col].replace('!@9#%8', np.nan)
        df_copy[col] = df_copy[col].replace('#F%$D@*&8', np.nan)
    
    # Convertendo colunas numéricas - primeiro limpando caracteres especiais
    numeric_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 
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

# Aplicando a limpeza aos conjuntos de dados
train_cleaned = clean_data(train_data)
test_cleaned = clean_data(test_data)

# Verificando valores nulos após limpeza
print("\nValores nulos após limpeza (treino):")
print(train_cleaned.isnull().sum())

# Separando features e target
X = train_cleaned.drop(['Credit_Score', 'ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan'], axis=1)
y = train_cleaned['Credit_Score']

# Categorizando features
categorical_features = ['Month', 'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Visualizando dados após a limpeza
print("\nFormato dos dados após limpeza:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

print("\nTipos de dados após limpeza:")
print(X.dtypes)

# %%
# Seção 5: Criação do Pipeline de Machine Learning
print("\nCriando pipeline de machine learning...")

# Preprocessador para features numéricas e categóricas
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Encoder para variável alvo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nClasses da variável alvo codificadas:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label}: {i}")

# %%
# Seção 6: Treinamento e otimização dos modelos
print("\nTreinando e otimizando modelos...")

# Divisão dos dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 1. Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parâmetros para a Grid Search (reduzidos para tornar o teste mais rápido)
rf_param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [10]
}

rf_grid_search = GridSearchCV(
    rf_pipeline, 
    rf_param_grid, 
    cv=3, 
    scoring='f1_weighted',
    n_jobs=-1, 
    verbose=1
)

print("\nTreinando Random Forest...")
rf_grid_search.fit(X_train, y_train)
print(f"Melhores parâmetros para Random Forest: {rf_grid_search.best_params_}")

# 2. XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
])

xgb_param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [5]
}

xgb_grid_search = GridSearchCV(
    xgb_pipeline, 
    xgb_param_grid, 
    cv=3, 
    scoring='f1_weighted',
    n_jobs=-1, 
    verbose=1
)

print("\nTreinando XGBoost...")
xgb_grid_search.fit(X_train, y_train)
print(f"Melhores parâmetros para XGBoost: {xgb_grid_search.best_params_}")

# 3. LightGBM
lgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(random_state=42))
])

lgb_param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [5]
}

lgb_grid_search = GridSearchCV(
    lgb_pipeline, 
    lgb_param_grid, 
    cv=3, 
    scoring='f1_weighted',
    n_jobs=-1, 
    verbose=1
)

print("\nTreinando LightGBM...")
lgb_grid_search.fit(X_train, y_train)
print(f"Melhores parâmetros para LightGBM: {lgb_grid_search.best_params_}")

# %%
# Seção 7: Avaliação dos modelos
print("\nAvaliando os modelos...")

# Função para avaliar e comparar modelos
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\nResultados para {model_name}:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"Precisão (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

# Avaliar cada modelo
results = []
results.append(evaluate_model(rf_grid_search, X_val, y_val, 'Random Forest'))
results.append(evaluate_model(xgb_grid_search, X_val, y_val, 'XGBoost'))
results.append(evaluate_model(lgb_grid_search, X_val, y_val, 'LightGBM'))

# Comparar resultados
results_df = pd.DataFrame(results)
print("\nComparação dos modelos:")
print(results_df)

# Encontrar o melhor modelo baseado no F1-Score
best_model_idx = results_df['f1_score'].argmax()
best_model_name = results_df.iloc[best_model_idx]['model_name']
print(f"\nMelhor modelo baseado no F1-Score: {best_model_name}")

# Selecionar o melhor modelo
if best_model_name == 'Random Forest':
    best_model = rf_grid_search
elif best_model_name == 'XGBoost':
    best_model = xgb_grid_search
else:
    best_model = lgb_grid_search

# %%
# Seção 8: Análise de importância das características
print("\nAnalisando importância das características...")

# Extrair o classificador do pipeline
best_classifier = best_model.named_steps['classifier']

# Obter nomes das features após o pré-processamento
preprocessor = best_model.named_steps['preprocessor']

# Verificar o tipo de modelo para extrair importância de características
if hasattr(best_classifier, 'feature_importances_'):
    # Obter nomes das features
    feature_names = []
    
    # Processar dados para obter as colunas transformadas
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Tentar extrair nomes de features do preprocessador
    try:
        # Para features numéricas
        num_features = preprocessor.transformers_[0][2]
        
        # Para features categóricas
        cat_features = preprocessor.transformers_[1][2]
        cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        
        # Ajustar o encoder para obter nomes das categorias
        cat_encoder.fit(X_train[cat_features])
        
        # Construir lista de nomes de features
        feature_names = list(num_features)
        
        # Adicionar nomes de features categóricas
        for i, cat in enumerate(cat_features):
            categories = cat_encoder.categories_[i]
            for category in categories:
                feature_names.append(f"{cat}_{category}")
    except:
        # Caso não consiga extrair nomes, usar índices
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
    
    # Obter importância das features
    importances = best_classifier.feature_importances_
    
    # Limitar ao tamanho da importância
    if len(feature_names) > len(importances):
        feature_names = feature_names[:len(importances)]
    elif len(feature_names) < len(importances):
        feature_names.extend([f"feature_{i}" for i in range(len(feature_names), len(importances))])
    
    # Criar DataFrame para visualizar importâncias
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Visualizar top 20 features mais importantes (ou menos se não houver 20)
    top_n = min(20, len(feature_importance_df))
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
    plt.title(f'Top {top_n} Features Mais Importantes - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nTop 10 características mais importantes:")
    print(feature_importance_df.head(10))
else:
    print("O modelo selecionado não suporta extração de importância de características.")

# %%
# Seção 9: Predições no conjunto de teste
print("\nRealizando predições no conjunto de teste...")

# Preparar o conjunto de teste
X_test = test_cleaned.drop(['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan'], axis=1)

# Fazer predições
test_predictions = best_model.predict(X_test)

# Decodificar predições para obter rótulos originais
test_predictions_decoded = label_encoder.inverse_transform(test_predictions)

# Criar DataFrame com resultados
test_results = pd.DataFrame({
    'ID': test_cleaned['ID'],
    'Credit_Score_Predicted': test_predictions_decoded
})

# Salvar predições
test_results.to_csv('credit_score_predictions.csv', index=False)
print("\nPredições salvas em 'credit_score_predictions.csv'")

# %%
# Seção 10: Conclusões e recomendações para decisões financeiras
print("\nConclusões e recomendações para decisões financeiras:")

print("""
CONCLUSÕES:

1. Modelo de Classificação:
   - O modelo de machine learning desenvolvido permite classificar clientes em três categorias de risco de crédito: 'Good', 'Standard' e 'Poor'.
   - Utilizamos F1-Score como métrica principal devido ao desbalanceamento das classes e importância igual de precisão e recall.

2. Aplicações Práticas:
   - Avaliação de risco para novos empréstimos: O modelo pode ajudar a instituição financeira a decidir se aprova ou não um empréstimo.
   - Definição de taxas de juros: Clientes classificados como 'Good' podem receber taxas mais baixas.
   - Definição de limites de crédito: O modelo pode ajudar a determinar limites de crédito apropriados para cada cliente.
   - Marketing direcionado: Ofertas personalizadas podem ser enviadas com base na classificação de crédito.

3. Principais fatores para um bom score de crédito:
   - Histórico de pagamento em dia
   - Maior tempo de histórico de crédito
   - Baixa utilização do crédito disponível
   - Diversidade de tipos de crédito (Credit_Mix)
   - Renda estável e compatível com os compromissos financeiros

4. Recomendações para melhorar o modelo:
   - Coletar mais dados para classes sub-representadas
   - Adicionar novas features como histórico de emprego e estabilidade residencial
   - Implementar técnicas de balanceamento de classes
   - Atualizar o modelo periodicamente com novos dados
""")

print("\nO trabalho foi concluído com sucesso!") 