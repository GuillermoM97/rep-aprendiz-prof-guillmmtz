# Inferencia ENR API on Cloud Run

Servicio de inferencia para estimar la probabilidad de **parto no institucional**
a partir de variables sociodemográficas y territoriales. El servicio está
montado sobre el flujo estándar de **Cloud Run** y expone una API FastAPI.

## Endpoints

- `GET /` — información básica del servicio
- `GET /health` — estado de carga de artefactos y baseline nacional
- `GET /ping/{ch}` — conectividad básica
- `POST /predict` — inferencia del modelo

## Request de ejemplo

```json
{
  "features": {
    "estado_ocurrencia": "San Luis Potosí",
    "tamano_localidad_ocurrencia": "De 1000 a 1999 habitantes",
    "edad_madre_rango": "21-25",
    "edad_madre_no_especificada": 0,
    "escolaridad_madre": "Secundaria o secundaria técnica completa",
    "condicion_actividad_madre": "No trabaja",
    "estado_civil_madre_modelo": "Unión libre",
    "edad_padre_rango": "21-25",
    "edad_padre_no_especificada": 0,
    "escolaridad_padre": "Secundaria o secundaria técnica completa",
    "condicion_actividad_padre": "Trabaja",
    "hijos_vivo_bucket": "1"
  }
}
```

## Respuesta de ejemplo

```json
{
  "prob_parto_no_institucional": 0.08,
  "riesgo_relativo_vs_nacional": 1.67,
  "latency_ms": 12.4
}
```

## Artefactos del modelo

El servicio carga estos archivos al arrancar:

- `enr_preprocess_ohe.joblib`
- `enr_xgb_cuda_booster.json`
- `enr_model_meta.json`

Puedes sobreescribir sus rutas mediante:

- `MODEL_PRE_PATH`
- `MODEL_BST_PATH`
- `META_PATH`

## Desarrollo local

Instala `invoke` y luego:

```bash
invoke setup-virtualenv
invoke dev
```

La API quedará disponible en `http://localhost:8080`.

## Tests

```bash
invoke test
```

Para system tests en Cloud Build:

```bash
gcloud builds submit \
  --config advance.cloudbuild.yaml \
  --substitutions 'COMMIT_SHA=manual,REPO_NAME=manual'
```

## Despliegue manual

```bash
export GOOGLE_CLOUD_PROJECT=<GCP_PROJECT_ID>
export REGION=us-central1
invoke build
invoke deploy
```

## Stack técnico

- FastAPI
- XGBoost
- scikit-learn
- Gunicorn + Uvicorn
- Cloud Run
- Cloud Build
