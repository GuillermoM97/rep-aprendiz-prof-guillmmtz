import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.logging import logger

MODEL_DIR = os.getenv("MODEL_DIR", "flan_t5_small_tass2020_mexico")
MODEL_GCS_URI = os.getenv(
    "MODEL_GCS_URI", "gs://projectdeeplearning/flan_t5_small_tass2020_mexico"
)
LOCAL_MODEL_CACHE_DIR = Path(
    os.getenv("LOCAL_MODEL_CACHE_DIR", "/tmp/flan_t5_small_tass2020_mexico")
)
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 8
PREFIX = (
    "Clasifica la polaridad del siguiente comentario en español de México. "
    "Responde solo con una palabra exacta de esta lista: "
    "positive, neutral, negative. Tweet: "
)

_torch = None
_tokenizer = None
_model = None
_device: Optional[str] = None
_resolved_model_dir: Optional[str] = None


def normalize_generated_label(text: str) -> str:
    z = str(text).strip().lower()
    z = z.replace(".", "").replace(",", "").replace(":", "").replace(";", "").strip()

    alias = {
        "positive": "positive",
        "positivo": "positive",
        "positiva": "positive",
        "pos": "positive",
        "p": "positive",
        "negative": "negative",
        "negativo": "negative",
        "negativa": "negative",
        "neg": "negative",
        "n": "negative",
        "neutral": "neutral",
        "neutro": "neutral",
        "neutra": "neutral",
        "neu": "neutral",
    }

    if z in alias:
        return alias[z]

    for k, v in alias.items():
        if k in z:
            return v

    return "neutral"


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise RuntimeError("MODEL_GCS_URI debe iniciar con gs://")

    path = uri[5:]
    bucket, _, prefix = path.partition("/")
    if not bucket or not prefix:
        raise RuntimeError("MODEL_GCS_URI debe incluir bucket y prefijo")
    return bucket, prefix.rstrip("/")


def _download_model_from_gcs(target_dir: Path) -> str:
    try:
        from google.cloud import storage
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Falta google-cloud-storage para descargar el modelo desde GCS."
        ) from exc

    bucket_name, prefix = _parse_gcs_uri(MODEL_GCS_URI)
    target_dir.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        raise RuntimeError(
            f"No se encontraron archivos del modelo en {MODEL_GCS_URI}"
        )

    logger.info(
        "model_download_started",
        model_gcs_uri=MODEL_GCS_URI,
        local_cache_dir=str(target_dir),
        bucket_name=bucket_name,
        prefix=prefix,
        blob_count=len(blobs),
    )

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = blob.name[len(prefix) :].lstrip("/")
        local_path = target_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)

    logger.info(
        "model_download_finished",
        model_gcs_uri=MODEL_GCS_URI,
        local_cache_dir=str(target_dir),
        downloaded_files=len([blob for blob in blobs if not blob.name.endswith("/")]),
    )
    return str(target_dir)


def _resolve_model_dir() -> str:
    global _resolved_model_dir

    if _resolved_model_dir is not None:
        return _resolved_model_dir

    local_dir = Path(MODEL_DIR)
    if local_dir.exists():
        logger.info("model_dir_resolved_local", model_dir=str(local_dir))
        _resolved_model_dir = str(local_dir)
        return _resolved_model_dir

    logger.info(
        "model_dir_resolved_gcs",
        model_dir=MODEL_DIR,
        model_gcs_uri=MODEL_GCS_URI,
        local_cache_dir=str(LOCAL_MODEL_CACHE_DIR),
    )
    _resolved_model_dir = _download_model_from_gcs(LOCAL_MODEL_CACHE_DIR)
    return _resolved_model_dir


def _load_model() -> None:
    global _torch, _tokenizer, _model, _device

    if _tokenizer is not None and _model is not None and _torch is not None:
        return

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Faltan dependencias del modelo. Instala torch, transformers y sentencepiece."
        ) from exc

    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
    resolved_model_dir = _resolve_model_dir()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("model_loading_started", model_dir=resolved_model_dir, device=_device)

    _tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_dir, local_files_only=True
    )
    logger.info("tokenizer_loaded", model_dir=resolved_model_dir)
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        resolved_model_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(_device)
    _model.eval()
    _torch = torch

    logger.info("model_loading_finished", model_dir=resolved_model_dir, device=_device)


def predict_sentiment(text: str) -> Dict[str, Any]:
    if not str(text).strip():
        raise ValueError("El texto no puede estar vacío")

    logger.info("prediction_requested", text_length=len(str(text)))
    _load_model()
    assert _torch is not None
    assert _tokenizer is not None
    assert _model is not None
    assert _device is not None

    with _torch.no_grad():
        enc = _tokenizer(
            PREFIX + str(text),
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LEN,
        )
        enc = {k: v.to(_device) for k, v in enc.items()}

        out = _model.generate(
            **enc,
            max_new_tokens=MAX_TARGET_LEN,
            do_sample=False,
        )

        raw = _tokenizer.decode(out[0], skip_special_tokens=True).strip()
        pred = normalize_generated_label(raw)

    return {
        "text": text,
        "raw_output": raw,
        "label": pred,
        "model_dir": _resolve_model_dir(),
        "device": _device,
    }


def get_model_debug_status() -> Dict[str, Any]:
    local_dir = Path(MODEL_DIR)
    cache_dir = LOCAL_MODEL_CACHE_DIR

    return {
        "model_dir_env": MODEL_DIR,
        "model_gcs_uri": MODEL_GCS_URI,
        "local_model_dir_exists": local_dir.exists(),
        "local_model_cache_dir": str(cache_dir),
        "local_model_cache_exists": cache_dir.exists(),
        "resolved_model_dir": _resolved_model_dir,
        "model_loaded": _model is not None,
        "tokenizer_loaded": _tokenizer is not None,
        "device": _device,
    }
