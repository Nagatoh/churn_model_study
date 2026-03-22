# Projeto de Churn

Este projeto saiu do estagio de notebook unico e agora tem uma base reproduzivel para:

- treinar modelos fora do notebook
- comparar variantes no MLflow
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

## Treino reproduzivel

O comando abaixo:

1. carrega o dataset
2. treina duas variantes da regressao logistica
3. registra as duas no MLflow
4. compara `with_tenure_group` vs `without_tenure_group`
5. promove uma delas para `artifacts/models/churn_model.joblib`
6. registra metricas de validacao cruzada no MLflow

```powershell
uv run churn-train
```

Por padrao, a variante promovida e `without_tenure_group`.

Se quiser promover a outra:

```powershell
uv run churn-train --selected-variant with_tenure_group
```

Se quiser apontar para outro CSV:

```powershell
uv run churn-train --dataset-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## O que fica versionado no MLflow

Cada run registra:

- variante do modelo
- hash `sha256` do dataset
- metadados do arquivo `.dvc` e commit Git
- dataset no campo nativo `Datasets` do MLflow
- metricas principais
- metricas medias e desvio padrao de validacao cruzada
- melhor threshold por `f1`
- figuras de avaliacao
- modelo `sklearn` logado no experimento

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
uv run churn-serve --model-path artifacts/models/with_tenure_group.joblib
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

1. incluir XGBoost ou LightGBM no mesmo fluxo de MLflow
2. adicionar testes de contrato para payloads invalidos e regressao de metricas
3. trocar o remote local do DVC por um storage compartilhado
4. colocar build e deploy automatizados para a API
