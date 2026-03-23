# Projeto de Churn

Este projeto saiu do estagio de notebook unico e agora tem uma base reproduzivel para:

- treinar o modelo campeao fora do notebook
- comparar variantes e tuning no fluxo de experimentacao
- validar o dataset com DVC e rastrear a versao usada no treino
- promover um modelo para producao local
- executar inferencia batch
- servir previsoes via API
- testar os componentes criticos em CI

O notebook continua util para exploracao em `notebooks/01_hypothesis_screening.ipynb`, mas o fluxo principal agora comeca em `src/`.

## Estrutura

- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: dataset base
- `src/churn_model/data.py`: carga, limpeza e preparacao
- `src/churn_model/training.py`: treino, avaliacao, MLflow e persistencia
- `src/churn_model/train.py`: CLI de treino
- `src/churn_model/experiment.py`: CLI de experimentacao com tuning
- `src/churn_model/inference.py`: reutilizacao do modelo promovido
- `src/churn_model/predict.py`: CLI de inferencia batch
- `src/churn_model/api.py`: API FastAPI
- `src/churn_model/serve.py`: CLI para subir a API
- `tests/`: testes minimos de transformacao, inferencia e API
- `.github/workflows/ci.yml`: pipeline de validacao no GitHub Actions
- `artifacts/models/`: modelo promovido para uso operacional
- `mlflow.db` e `mlartifacts/`: tracking local do MLflow
- `data/*.dvc`: ponteiro DVC para a versao do dataset

## Setup com uv

```powershell
uv sync
```

Se quiser abrir o notebook:

```powershell
uv run jupyter lab
```

## Treino operacional

O comando abaixo:

1. carrega o dataset
2. treina apenas a variante campea congelada
3. registra a run no MLflow
4. promove o artifact para `artifacts/models/churn_model.joblib`

```powershell
uv run churn-train
```

Hoje o treino operacional usa a variante fixa `xgboost_with_tenure_group`, com os melhores hiperparametros encontrados no ciclo de experimentacao.

## Experimentos e tuning

O comando abaixo roda o fluxo mais pesado de comparacao:

1. baseline linear com regularizacao e `GridSearchCV`
2. `XGBoost` com `RandomizedSearchCV`
3. comparacao entre familias de modelo e variacoes com ou sem `tenure_group`
4. promocao automatica da melhor run experimental

```powershell
uv run churn-experiment
```

Por padrao, a melhor variante experimental e promovida usando `cv_f1_mean`.

Se quiser trocar a metrica de promocao automatica:

```powershell
uv run churn-experiment --selection-metric roc_auc
```

Se quiser forcar manualmente uma variante:

```powershell
uv run churn-experiment --selected-variant xgboost_without_tenure_group
```

Se quiser apontar para outro CSV:

```powershell
uv run churn-train --dataset-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## O que fica versionado no MLflow

Cada run registra:

- variante do modelo
- familia do modelo
- estrategia de tuning usada
- melhores hiperparametros encontrados
- penalidade de regularizacao selecionada no baseline linear
- contagem de features selecionadas via metodo embutido
- hash `sha256` do dataset
- metadados do arquivo `.dvc` e commit Git
- dataset no campo nativo `Datasets` do MLflow
- metricas principais
- `tuning_best_cv_f1` e `tuning_best_cv_roc_auc`
- metricas medias e desvio padrao de validacao cruzada
- melhor threshold por `f1`
- figuras de avaliacao
- modelo `sklearn` logado no experimento

O artifact promovido para `artifacts/models/churn_model.joblib` tambem recebe metadados de promocao em `artifacts/models/churn_model_metadata.json`, incluindo:

- modo de promocao: automatico ou manual
- metrica usada na selecao
- variante promovida
- timestamp da promocao

MLflow e DVC se complementam aqui:

- DVC controla a versao do dataset
- MLflow guarda experimento, metricas, artefatos e a referencia ao dataset usado

## MLflow local

O projeto usa:

- backend SQLite em `mlflow.db`
- artefatos em `mlartifacts/`

Para abrir a UI:

```powershell
uv run python -m mlflow server --backend-store-uri sqlite:///mlflow.db
```

Depois abra `http://127.0.0.1:5000`.

## Inferencia batch

Depois de promover um modelo, rode inferencia em lote a partir de um CSV com as features cruas do cliente.

Exemplo:

```powershell
uv run churn-predict --input-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-path predictions.json
```

Observacao:

- o script espera as colunas de entrada cruas do cliente
- `customerID` e `Churn` nao sao necessarios
- o preprocessamento correto e reaplicado automaticamente

## Testes

Para validar os componentes principais localmente:

```powershell
uv run pytest
```

Os testes cobrem:

- preparacao de dados
- inferencia com artifact promovido
- endpoints principais da API
- contrato de payload invalido
- piso minimo de metricas para `LogisticRegression` e `XGBoost`
- retorno do tuning e da selecao embutida de variaveis

## API de previsao

Para subir a API local:

```powershell
uv run churn-serve
```

Endpoints:

- `GET /health`
- `POST /predict`

A API usa por padrao `artifacts/models/churn_model.joblib`.

Se quiser apontar para outro artifact:

```powershell
uv run churn-serve --model-path artifacts/models/xgboost_without_tenure_group.joblib
```

## Docker

Existe um `Dockerfile` para servir a API.

Build:

```powershell
docker build -t churn-model .
```

Run:

```powershell
docker run --rm -p 8000:8000 -v ${PWD}/artifacts:/app/artifacts churn-model
```

Importante:

- o container nao treina modelo
- ele espera que o artifact promovido ja exista em `artifacts/models/churn_model.joblib`

## Proximos passos recomendados

Com essa base pronta, a sequencia mais natural e:

1. adicionar `LightGBM` como terceira familia no mesmo fluxo de tuning
2. trocar o remote local do DVC por um storage compartilhado
3. colocar build e deploy automatizados para a API
