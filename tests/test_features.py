import pandas as pd

from src.features import add_features


def sample():
    return pd.DataFrame({"valor_transacao": [200.0], "valor_medio_historico": [100.0], "hora_transacao": [3], "dia_semana": [6]})


def test_adds_domain_features():
    result = add_features(sample())
    assert result.loc[0, "razao_valor_medio"] == 2
    assert result.loc[0, "horario_suspeito"] == 1
    assert result.loc[0, "fim_de_semana"] == 1


def test_does_not_mutate_input():
    data = sample()
    add_features(data)
    assert "razao_valor_medio" not in data


def test_historical_value_zero_is_safe():
    data = sample()
    data["valor_medio_historico"] = 0
    assert add_features(data).loc[0, "razao_valor_medio"] == 200
