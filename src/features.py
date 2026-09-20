import pandas as pd


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic domain features without mutating the input."""
    result = data.copy()
    result["razao_valor_medio"] = result["valor_transacao"] / result["valor_medio_historico"].clip(lower=1)
    result["horario_suspeito"] = result["hora_transacao"].between(0, 6).astype(int)
    result["fim_de_semana"] = result["dia_semana"].ge(5).astype(int)
    return result
