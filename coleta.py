"""
==============================================================================
COLETA E PREPARAÇÃO DE DADOS PARA DETECÇÃO DE FRAUDES
==============================================================================

Este script implementa a primeira etapa do pipeline de detecção de fraudes:
- Geração de dados sintéticos balanceados
- Features relevantes para detecção de fraude
- Análise exploratória dos dados
- Preparação para modelagem

Autor: Sistema de Detecção de Fraudes
==============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configurações
np.random.seed(42)
sns.set_style('whitegrid')

print("=" * 80)
print("SISTEMA DE DETECÇÃO DE FRAUDES - COLETA DE DADOS")
print("=" * 80)
print()

# 1. GERAÇÃO DE DADOS SINTÉTICOS
def gerar_dados_transacoes(n_samples=100000, fraude_ratio=0.01):
    """
    Gera dados sintéticos de transações financeiras
    
    Args:
        n_samples: número total de transações
        fraude_ratio: proporção de fraudes (padrão: 1%)
    
    Returns:
        DataFrame com transações
    """
    print(f"Gerando {n_samples:,} transações (fraude: {fraude_ratio*100:.1f}%)")
    print("-" * 80)
    
    # Número de fraudes
    n_fraudes = int(n_samples * fraude_ratio)
    n_normais = n_samples - n_fraudes
    
    # Criar array de labels
    labels = np.concatenate([
        np.zeros(n_normais),  # Transações normais
        np.ones(n_fraudes)     # Fraudes
    ])
    
    # Inicializar DataFrame
    data = pd.DataFrame()
    data['is_fraud'] = labels
    
    # --- Feature 1: Valor da Transação ---
    # Transações normais: distribuição log-normal (valores típicos)
    # Fraudes: valores mais altos e distribuição diferente
    valor_normal = np.random.lognormal(mean=4.5, sigma=1.2, size=n_normais)
    valor_fraude = np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraudes)
    
    data['valor_transacao'] = np.concatenate([valor_normal, valor_fraude])
    data['valor_transacao'] = np.clip(data['valor_transacao'], 1, 10000)
    
    # --- Feature 2: Hora do dia ---
    # Normais: distribuição normal durante horário comercial
    # Fraudes: mais comuns em horários incomuns (madrugada)
    hora_normal = np.random.normal(loc=14, scale=4, size=n_normais) % 24
    hora_fraude = np.concatenate([
        np.random.normal(loc=3, scale=2, size=n_fraudes//2) % 24,  # Madrugada
        np.random.normal(loc=22, scale=2, size=n_fraudes//2) % 24  # Noite
    ])
    
    data['hora_transacao'] = np.concatenate([hora_normal, hora_fraude])
    data['hora_transacao'] = np.clip(data['hora_transacao'], 0, 23).astype(int)
    
    # --- Feature 3: Dia da semana ---
    # 0=Segunda, 6=Domingo
    dia_normal = np.random.choice(range(7), size=n_normais, p=[0.16, 0.16, 0.16, 0.16, 0.16, 0.1, 0.1])
    dia_fraude = np.random.choice(range(7), size=n_fraudes, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25])
    
    data['dia_semana'] = np.concatenate([dia_normal, dia_fraude])
    
    # --- Feature 4: Distância da última transação (km) ---
    # Normais: próximas
    # Fraudes: distâncias incomuns
    dist_normal = np.abs(np.random.normal(loc=5, scale=10, size=n_normais))
    dist_fraude = np.abs(np.random.normal(loc=150, scale=200, size=n_fraudes))
    
    data['distancia_ultima_transacao'] = np.concatenate([dist_normal, dist_fraude])
    data['distancia_ultima_transacao'] = np.clip(data['distancia_ultima_transacao'], 0, 1000)
    
    # --- Feature 5: Número de transações nas últimas 24h ---
    # Normais: poucas transações
    # Fraudes: tentativas múltiplas
    trans_normal = np.random.poisson(lam=2, size=n_normais)
    trans_fraude = np.random.poisson(lam=8, size=n_fraudes)
    
    data['num_transacoes_24h'] = np.concatenate([trans_normal, trans_fraude])
    data['num_transacoes_24h'] = np.clip(data['num_transacoes_24h'], 0, 50)
    
    # --- Feature 6: Idade da conta (dias) ---
    # Normais: contas mais antigas
    # Fraudes: contas novas
    idade_normal = np.abs(np.random.normal(loc=800, scale=400, size=n_normais))
    idade_fraude = np.abs(np.random.normal(loc=30, scale=50, size=n_fraudes))
    
    data['idade_conta_dias'] = np.concatenate([idade_normal, idade_fraude])
    data['idade_conta_dias'] = np.clip(data['idade_conta_dias'], 1, 3650)
    
    # --- Feature 7: Tipo de cartão ---
    # 0=Débito, 1=Crédito, 2=Pré-pago
    tipo_normal = np.random.choice([0, 1, 2], size=n_normais, p=[0.4, 0.5, 0.1])
    tipo_fraude = np.random.choice([0, 1, 2], size=n_fraudes, p=[0.2, 0.5, 0.3])
    
    data['tipo_cartao'] = np.concatenate([tipo_normal, tipo_fraude])
    
    # --- Feature 8: Canal de transação ---
    # 0=Online, 1=Físico, 2=ATM, 3=App
    canal_normal = np.random.choice([0, 1, 2, 3], size=n_normais, p=[0.3, 0.4, 0.1, 0.2])
    canal_fraude = np.random.choice([0, 1, 2, 3], size=n_fraudes, p=[0.6, 0.1, 0.05, 0.25])
    
    data['canal_transacao'] = np.concatenate([canal_normal, canal_fraude])
    
    # --- Feature 9: País da transação (0=doméstico, 1=internacional) ---
    pais_normal = np.random.choice([0, 1], size=n_normais, p=[0.95, 0.05])
    pais_fraude = np.random.choice([0, 1], size=n_fraudes, p=[0.6, 0.4])
    
    data['transacao_internacional'] = np.concatenate([pais_normal, pais_fraude])
    
    # --- Feature 10: Falhas de autenticação nas últimas 24h ---
    falhas_normal = np.random.poisson(lam=0.1, size=n_normais)
    falhas_fraude = np.random.poisson(lam=3, size=n_fraudes)
    
    data['falhas_autenticacao_24h'] = np.concatenate([falhas_normal, falhas_fraude])
    data['falhas_autenticacao_24h'] = np.clip(data['falhas_autenticacao_24h'], 0, 20)
    
    # --- Feature 11: Categoria do comerciante ---
    # 0=Supermercado, 1=Restaurante, 2=Eletrônicos, 3=Vestuário, 4=Combustível, 5=Outros
    cat_normal = np.random.choice(range(6), size=n_normais, p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])
    cat_fraude = np.random.choice(range(6), size=n_fraudes, p=[0.05, 0.05, 0.4, 0.25, 0.05, 0.2])
    
    data['categoria_comerciante'] = np.concatenate([cat_normal, cat_fraude])
    
    # --- Feature 12: Valor médio das últimas transações ---
    valor_medio_normal = np.random.lognormal(mean=4.2, sigma=1.0, size=n_normais)
    valor_medio_fraude = np.random.lognormal(mean=4.0, sigma=1.5, size=n_fraudes)
    
    data['valor_medio_historico'] = np.concatenate([valor_medio_normal, valor_medio_fraude])
    data['valor_medio_historico'] = np.clip(data['valor_medio_historico'], 1, 5000)
    
    # --- Feature Derivada: Razão valor atual / valor médio ---
    data['razao_valor_medio'] = data['valor_transacao'] / (data['valor_medio_historico'] + 1)
    
    # --- Feature Derivada: Flag horário suspeito (0-6h) ---
    data['horario_suspeito'] = ((data['hora_transacao'] >= 0) & (data['hora_transacao'] <= 6)).astype(int)
    
    # --- Feature Derivada: Flag fim de semana ---
    data['fim_de_semana'] = (data['dia_semana'] >= 5).astype(int)
    
    # Embaralhar dados
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Adicionar ID único
    data.insert(0, 'transaction_id', range(1, len(data) + 1))
    
    # Adicionar timestamp
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(seconds=np.random.randint(0, 30*24*3600)) for _ in range(len(data))]
    data.insert(1, 'timestamp', timestamps)
    
    print(f"Dataset gerado com sucesso!")
    print(f"   - Total de transações: {len(data):,}")
    print(f"   - Transações normais: {(data['is_fraud']==0).sum():,} ({(data['is_fraud']==0).sum()/len(data)*100:.2f}%)")
    print(f"   - Fraudes: {(data['is_fraud']==1).sum():,} ({(data['is_fraud']==1).sum()/len(data)*100:.2f}%)")
    print(f"   - Número de features: {len(data.columns)-3}")  # -3 (id, timestamp, target)
    print()
    
    return data



# 2. ANÁLISE EXPLORATÓRIA DOS DADOS

def analisar_dados(df):
    """
    Realiza análise exploratória dos dados
    """
    print("=" * 80)
    print(" ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 80)
    print()
    
    # Informações gerais
    print(" Informações Gerais do Dataset:")
    print("-" * 80)
    print(df.info())
    print()
    
    # Estatísticas descritivas
    print(" Estatísticas Descritivas:")
    print("-" * 80)
    print(df.describe())
    print()
    
    # Análise de valores faltantes
    print(" Análise de Valores Faltantes:")
    print("-" * 80)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(" Nenhum valor faltante encontrado!")
    else:
        print(missing[missing > 0])
    print()
    
    # Distribuição da variável target
    print(" Distribuição da Variável Target (is_fraud):")
    print("-" * 80)
    fraud_dist = df['is_fraud'].value_counts()
    fraud_pct = df['is_fraud'].value_counts(normalize=True) * 100
    
    print(f"Classe 0 (Normal): {fraud_dist[0]:,} ({fraud_pct[0]:.2f}%)")
    print(f"Classe 1 (Fraude): {fraud_dist[1]:,} ({fraud_pct[1]:.2f}%)")
    print()
    
    # Comparação de médias entre classes
    print(" Comparação de Médias entre Transações Normais e Fraudulentas:")
    print("-" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['transaction_id', 'is_fraud'])
    
    comparison = pd.DataFrame({
        'Normal': df[df['is_fraud']==0][numeric_cols].mean(),
        'Fraude': df[df['is_fraud']==1][numeric_cols].mean()
    })
    comparison['Diferença (%)'] = ((comparison['Fraude'] - comparison['Normal']) / comparison['Normal'] * 100).round(2)
    
    print(comparison.round(2))
    print()
    
    # Correlação com target
    print(" Correlação das Features com is_fraud:")
    print("-" * 80)
    correlations = df[numeric_cols].corrwith(df['is_fraud']).sort_values(ascending=False)
    print(correlations.round(4))
    print()
    
    return comparison, correlations



# 3. PREPARAÇÃO DOS DADOS PARA MODELAGEM

def preparar_dados_modelagem(df, test_size=0.2, val_size=0.1):
    """
    Prepara dados para modelagem com split treino/validação/teste
    """
    print("=" * 80)
    print(" PREPARAÇÃO DOS DADOS PARA MODELAGEM")
    print("=" * 80)
    print()
    
    # Separar features e target
    X = df.drop(['transaction_id', 'timestamp', 'is_fraud'], axis=1)
    y = df['is_fraud']
    
    print(" Shape dos dados:")
    print(f"   - Features (X): {X.shape}")
    print(f"   - Target (y): {y.shape}")
    print()
    
    # Split treino/temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42, stratify=y
    )
    
    # Split validação/teste
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size + val_size), 
        random_state=42, stratify=y_temp
    )
    
    print(f" Divisão dos dados:")
    print(f"   - Treino: {len(X_train):,} amostras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"     • Normal: {(y_train==0).sum():,} | Fraude: {(y_train==1).sum():,}")
    print(f"   - Validação: {len(X_val):,} amostras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"     • Normal: {(y_val==0).sum():,} | Fraude: {(y_val==1).sum():,}")
    print(f"   - Teste: {len(X_test):,} amostras ({len(X_test)/len(X)*100:.1f}%)")
    print(f"     • Normal: {(y_test==0).sum():,} | Fraude: {(y_test==1).sum():,}")
    print()
    
    # Normalização
    print(" Normalizando features numéricas...")
    scaler = StandardScaler()
    
    # Fit apenas no treino
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Converter de volta para DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(" Normalização concluída!")
    print()
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }



# 4. SALVAR DADOS

def salvar_dados(df, dados_preparados, output_dir='data'):
    """
    Salva os dados processados
    """
    import os
    
    print("=" * 80)
    print(" SALVANDO DADOS")
    print("=" * 80)
    print()
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar dataset completo
    df.to_csv(f'{output_dir}/transacoes_completo.csv', index=False)
    print(f" Dataset completo salvo: {output_dir}/transacoes_completo.csv")
    
    # Salvar datasets de treino/val/teste
    dados_preparados['X_train'].to_csv(f'{output_dir}/X_train.csv', index=False)
    dados_preparados['X_val'].to_csv(f'{output_dir}/X_val.csv', index=False)
    dados_preparados['X_test'].to_csv(f'{output_dir}/X_test.csv', index=False)
    
    dados_preparados['y_train'].to_csv(f'{output_dir}/y_train.csv', index=False)
    dados_preparados['y_val'].to_csv(f'{output_dir}/y_val.csv', index=False)
    dados_preparados['y_test'].to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print(f" Dados de treino salvos: {output_dir}/X_train.csv, y_train.csv")
    print(f" Dados de validação salvos: {output_dir}/X_val.csv, y_val.csv")
    print(f" Dados de teste salvos: {output_dir}/X_test.csv, y_test.csv")
    print()
    
    # Salvar scaler
    import joblib
    joblib.dump(dados_preparados['scaler'], f'{output_dir}/scaler.pkl')
    print(f" Scaler salvo: {output_dir}/scaler.pkl")
    print()



# 5. VISUALIZAÇÕES

def criar_visualizacoes(df, output_dir='data'):
    """
    Cria visualizações dos dados
    """
    import os
    
    print("=" * 80)
    print(" CRIANDO VISUALIZAÇÕES")
    print("=" * 80)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribuição de valores por classe
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Valor da transação
    axes[0, 0].hist(df[df['is_fraud']==0]['valor_transacao'], bins=50, alpha=0.7, label='Normal', color='blue')
    axes[0, 0].hist(df[df['is_fraud']==1]['valor_transacao'], bins=50, alpha=0.7, label='Fraude', color='red')
    axes[0, 0].set_xlabel('Valor da Transação')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].set_title('Distribuição: Valor da Transação')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Hora da transação
    axes[0, 1].hist(df[df['is_fraud']==0]['hora_transacao'], bins=24, alpha=0.7, label='Normal', color='blue')
    axes[0, 1].hist(df[df['is_fraud']==1]['hora_transacao'], bins=24, alpha=0.7, label='Fraude', color='red')
    axes[0, 1].set_xlabel('Hora do Dia')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].set_title('Distribuição: Hora da Transação')
    axes[0, 1].legend()
    
    # Distância
    axes[1, 0].hist(df[df['is_fraud']==0]['distancia_ultima_transacao'], bins=50, alpha=0.7, label='Normal', color='blue')
    axes[1, 0].hist(df[df['is_fraud']==1]['distancia_ultima_transacao'], bins=50, alpha=0.7, label='Fraude', color='red')
    axes[1, 0].set_xlabel('Distância (km)')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Distribuição: Distância da Última Transação')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # Transações 24h
    axes[1, 1].hist(df[df['is_fraud']==0]['num_transacoes_24h'], bins=30, alpha=0.7, label='Normal', color='blue')
    axes[1, 1].hist(df[df['is_fraud']==1]['num_transacoes_24h'], bins=30, alpha=0.7, label='Fraude', color='red')
    axes[1, 1].set_xlabel('Número de Transações')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição: Transações nas Últimas 24h')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distribuicoes.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: {output_dir}/distribuicoes.png")
    
    # 2. Matriz de correlação
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['transaction_id'])
    corr_matrix = df[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlação das Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlacao.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: {output_dir}/correlacao.png")
    
    # 3. Box plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    features_to_plot = ['valor_transacao', 'distancia_ultima_transacao', 
                        'num_transacoes_24h', 'idade_conta_dias', 
                        'falhas_autenticacao_24h', 'razao_valor_medio']
    
    for i, feature in enumerate(features_to_plot):
        df.boxplot(column=feature, by='is_fraud', ax=axes[i])
        axes[i].set_title(f'{feature}')
        axes[i].set_xlabel('is_fraud (0=Normal, 1=Fraude)')
        axes[i].set_ylabel(feature)
    
    plt.suptitle('Comparação de Features: Normal vs Fraude', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplots.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico salvo: {output_dir}/boxplots.png")
    print()



# 6. EXECUTAR PIPELINE COMPLETO

def executar_pipeline_completo():
    """
    Executa todo o pipeline de coleta e preparação de dados
    """
    print()
    print(" Iniciando Pipeline de Coleta e Preparação de Dados")
    print()
    
    # 1. Gerar dados
    df = gerar_dados_transacoes(n_samples=100000, fraude_ratio=0.01)
    
    # 2. Análise exploratória
    comparison, correlations = analisar_dados(df)
    
    # 3. Preparar dados para modelagem
    dados_preparados = preparar_dados_modelagem(df)
    
    # 4. Criar visualizações
    criar_visualizacoes(df)
    
    # 5. Salvar dados
    salvar_dados(df, dados_preparados)
    
    print("=" * 80)
    print(" PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print()
    print(" Arquivos gerados:")
    print("   - data/transacoes_completo.csv")
    print("   - data/X_train.csv, y_train.csv")
    print("   - data/X_val.csv, y_val.csv")
    print("   - data/X_test.csv, y_test.csv")
    print("   - data/scaler.pkl")
    print("   - data/distribuicoes.png")
    print("   - data/correlacao.png")
    print("   - data/boxplots.png")
    print()
    print(" Próxima etapa: Treinamento do modelo (Etapa 3)")
    print()
    
    return df, dados_preparados



# EXECUÇÃO
if __name__ == "__main__":
    df, dados_preparados = executar_pipeline_completo()
    
    # Exibir primeiras linhas
    print("=" * 80)
    print(" PREVIEW DOS DADOS GERADOS")
    print("=" * 80)
    print()
    print(df.head(10))
    print()
    print(f" Features disponíveis: {', '.join(dados_preparados['feature_names'])}")