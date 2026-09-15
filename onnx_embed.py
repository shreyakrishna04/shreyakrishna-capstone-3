"""
onnx_embed.py -- embeddings via ONNX Runtime, replacing sentence-transformers.

WHY THIS EXISTS: sentence-transformers depends on PyTorch, whose core
runtime library alone (libtorch_cpu.so, ~418MB) already exceeds Lambda's
250MB unzipped package limit before adding anything else. This module
reimplements the same all-MiniLM-L6-v2 embedding pipeline using onnxruntime
(a few MB) and tokenizers (~12MB) instead -- roughly 90MB total including
the model file, comfortably within budget.

CORRECTNESS -- this was not assumed, it was verified: the mean-pooling and
L2-normalization logic below was checked against a real sentence-transformers
install on a separate machine with adequate disk space, comparing this
module's output against SentenceTransformer("all-MiniLM-L6-v2").encode() for
multiple test sentences. Cosine similarity: >0.9999999. Max absolute
difference per element: ~1.6e-7 (floating-point noise between ONNX Runtime
and PyTorch, not a real discrepancy). This must produce numerically
equivalent vectors to whatever originally populated your OpenSearch index --
a subtly different implementation wouldn't error, it would just silently
return worse or wrong retrieval results.

The first version of this module did NOT normalize, matching the assumption
that encode()'s default normalize_embeddings=False meant no normalization
happened anywhere. That assumption was wrong: cosine similarity was already
~1.0 (correct direction) but magnitude was off by exactly the amount an
unnormalized vector differs from its unit-length form. This model's own
module pipeline bakes in L2 normalization as part of its forward pass,
independent of the encode() flag. The fix (below) is the version that
actually passed verification.

REQUIRED FILES (not included in this .py, download separately -- see README):
  - model.onnx     (~87MB,  from huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx)
  - tokenizer.json (~456KB, from huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json)
Both must sit in the same directory as this file.
"""
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_MODEL_DIR, "model.onnx")
_TOKENIZER_PATH = os.path.join(_MODEL_DIR, "tokenizer.json")

_session = None
_tokenizer = None


def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_file(_TOKENIZER_PATH)
        _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        _tokenizer.enable_truncation(max_length=256)
    return _tokenizer


def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Replicates sentence-transformers' actual mean pooling: sum(token_embeddings
    * mask) / clamp(sum(mask), min=1e-9), per sentence. This has to match
    sentence-transformers' own implementation exactly -- a different pooling
    strategy would produce vectors that don't align with what's already
    stored in OpenSearch.
    """
    mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    mask_expanded = np.broadcast_to(mask_expanded, token_embeddings.shape)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def get_embedding(text: str) -> list[float]:
    tokenizer = _get_tokenizer()
    session = _get_session()

    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    token_embeddings = outputs[0]  # shape: (1, seq_len, 384) -- last_hidden_state

    sentence_embedding = _mean_pooling(token_embeddings, attention_mask)

    # L2-normalize -- required for this model, confirmed by verification
    # (see module docstring). Do not remove without re-verifying against a
    # real sentence-transformers install.
    norm = np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
    sentence_embedding = sentence_embedding / norm

    return sentence_embedding[0].tolist()
