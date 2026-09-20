from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["valor_transacao", "hora_transacao", "distancia_ultima_transacao", "num_transacoes_24h", "idade_conta_dias", "falhas_autenticacao_24h", "valor_medio_historico", "razao_valor_medio"]
CATEGORICAL_FEATURES = ["dia_semana", "tipo_cartao", "canal_transacao", "transacao_internacional", "categoria_comerciante", "horario_suspeito", "fim_de_semana"]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)])
