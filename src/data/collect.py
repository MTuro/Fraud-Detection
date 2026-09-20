from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, TARGET

RAW_FEATURES = [
    "valor_transacao", "hora_transacao", "dia_semana", "distancia_ultima_transacao",
    "num_transacoes_24h", "idade_conta_dias", "tipo_cartao", "canal_transacao",
    "transacao_internacional", "falhas_autenticacao_24h", "categoria_comerciante",
    "valor_medio_historico",
]
REQUIRED_COLUMNS = ["transaction_id", "timestamp", *RAW_FEATURES, TARGET]


def generate_transactions(n_samples: int = 50_000, fraud_ratio: float = 0.01, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a reproducible, imbalanced synthetic transaction dataset."""
    if n_samples < 100 or not 0 < fraud_ratio < 0.5:
        raise ValueError("n_samples must be >= 100 and fraud_ratio between 0 and 0.5")
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, round(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud
    data = pd.DataFrame({
        "valor_transacao": np.r_[rng.lognormal(4.5, 1.2, n_normal), rng.lognormal(5.0, 1.4, n_fraud)].clip(1, 10_000),
        "hora_transacao": np.r_[rng.normal(14, 4, n_normal) % 24, rng.normal(12, 6, n_fraud) % 24].astype(int),
        "dia_semana": np.r_[rng.choice(7, n_normal, p=[.16, .16, .16, .16, .16, .10, .10]), rng.choice(7, n_fraud, p=[.13, .13, .13, .13, .13, .175, .175])],
        "distancia_ultima_transacao": np.r_[abs(rng.normal(5, 10, n_normal)), abs(rng.normal(50, 80, n_fraud))].clip(0, 1_000),
        "num_transacoes_24h": np.r_[rng.poisson(2, n_normal), rng.poisson(4, n_fraud)].clip(0, 50),
        "idade_conta_dias": np.r_[abs(rng.normal(800, 400, n_normal)), abs(rng.normal(400, 350, n_fraud))].clip(1, 3_650),
        "tipo_cartao": np.r_[rng.choice(3, n_normal, p=[.4, .5, .1]), rng.choice(3, n_fraud, p=[.3, .5, .2])],
        "canal_transacao": np.r_[rng.choice(4, n_normal, p=[.3, .4, .1, .2]), rng.choice(4, n_fraud, p=[.45, .25, .075, .225])],
        "transacao_internacional": np.r_[rng.choice(2, n_normal, p=[.95, .05]), rng.choice(2, n_fraud, p=[.775, .225])],
        "falhas_autenticacao_24h": np.r_[rng.poisson(.1, n_normal), rng.poisson(1, n_fraud)].clip(0, 20),
        "categoria_comerciante": np.r_[rng.choice(6, n_normal, p=[.25, .20, .15, .15, .15, .10]), rng.choice(6, n_fraud, p=[.15, .125, .275, .20, .10, .15])],
        "valor_medio_historico": np.r_[rng.lognormal(4.2, 1.0, n_normal), rng.lognormal(4.1, 1.3, n_fraud)].clip(1, 5_000),
        TARGET: np.r_[np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)],
    })
    n_noisy = round(n_fraud * .10)
    if n_noisy:
        normal_noise = rng.choice(n_normal, n_noisy, replace=False)
        fraud_noise = rng.choice(np.arange(n_normal, n_samples), n_noisy, replace=False)
        data.loc[normal_noise, TARGET] = 1
        data.loc[fraud_noise, TARGET] = 0
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    data.insert(0, "transaction_id", np.arange(1, n_samples + 1))
    start = pd.Timestamp("2024-01-01", tz="UTC")
    data.insert(1, "timestamp", start + pd.to_timedelta(rng.integers(0, 30 * 86_400, n_samples), unit="s"))
    validate_transactions(data)
    return data


def validate_transactions(data: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[["transaction_id", "timestamp", TARGET]].isna().any().any():
        raise ValueError("Identifiers, timestamps, and target cannot be missing")
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1")


def load_transactions(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["timestamp"])
    validate_transactions(data)
    return data
