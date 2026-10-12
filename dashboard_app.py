"""Local, read-only visual dashboard for the two fraud experiments.

Run ``python dashboard_app.py`` and open http://127.0.0.1:8765.
The app serves only curated local summaries; it never exposes raw CSV files.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

import numpy as np
from sklearn.metrics import precision_recall_curve

from dashboard_data import CACHE_DIR, DATASETS, OUTPUTS, build_dataset_cache


ASSETS = Path(__file__).resolve().parent / "dashboard"
RUN_PATTERN = re.compile(r"^(validation_v2|hybrid_search|feature_search|minibatch_search|thesis_v1)_\d{8}_\d{6}$")
MODEL_FILE = {"baseline": "baseline", "autoencoder": "autoencoder", "gnn": "gnn", "hybrid": "hybrid"}


def _run_kind(name: str) -> str:
    if name.startswith("thesis_v1_"):
        return "thesis"
    if name.startswith("validation_v2_"):
        return "comparison"
    if name.startswith("hybrid_search_"):
        return "hybrid"
    if name.startswith("minibatch_search_"):
        return "minibatch"
    return "feature_selection"


def _feature_results_file(directory: Path) -> Path:
    """Read new exploratory runs and the earlier, misnamed saved run."""
    current = directory / "exploratory_test.json"
    return current if current.is_file() else directory / "validation_only.json"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_json(value):
    """Keep historical non-finite thresholds from breaking the JSON API."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def list_runs() -> list[dict]:
    runs = []
    if not OUTPUTS.is_dir():
        return runs
    for directory in sorted(OUTPUTS.iterdir(), key=lambda path: path.name[-15:], reverse=True):
        if not directory.is_dir() or not RUN_PATTERN.fullmatch(directory.name):
            continue
        kind = _run_kind(directory.name)
        if kind == "thesis":
            result_file = directory / "results.json"
            if not result_file.is_file():
                continue
            protocol, rows = _json(directory / "protocol.json"), _json(result_file)
            ready = [dataset for dataset in protocol["datasets"] if any(row["dataset"] == dataset for row in rows)]
            runs.append({"id": directory.name, "kind": kind, "ready_datasets": ready,
                         "complete": protocol["test_evaluated"], "shap_complete": protocol["shap_complete"]})
            continue
        if kind == "minibatch":
            result_file = directory / "results.json"
            protocol_file = directory / "protocol.json"
            if not result_file.is_file() or not protocol_file.is_file():
                continue
            rows, protocol = _json(result_file), _json(protocol_file)
            ready = [dataset for dataset in DATASETS if any(row["dataset"] == dataset for row in rows)]
            if ready:
                complete = all(any(row["dataset"] == dataset and row["seed"] == seed and row["model"] == model for row in rows)
                               for dataset in protocol["datasets"] for seed in protocol["seeds"] for model in ("hybrid", "baseline"))
                runs.append({"id": directory.name, "kind": kind, "ready_datasets": ready, "complete": complete})
            continue
        if kind == "feature_selection":
            protocol_file = directory / "protocol.json"
            results_file = _feature_results_file(directory)
            if not protocol_file.is_file() or not results_file.is_file():
                continue
            protocol = _json(protocol_file)
            rows = _json(results_file)
            if not protocol.get("test_evaluated") or not rows or any("metrics" not in row for row in rows):
                continue
            dataset = protocol.get("dataset")
            if dataset in DATASETS:
                runs.append({"id": directory.name, "kind": kind, "ready_datasets": [dataset],
                             "complete": len(rows) == len(protocol.get("seeds", []))})
            continue
        ready = []
        for dataset in DATASETS:
            source = directory / "results.json" if kind == "comparison" else directory / dataset / "selected_test.json"
            if source.is_file():
                ready.append(dataset)
        runs.append({"id": directory.name, "kind": kind, "ready_datasets": ready,
                     "complete": len(ready) == len(DATASETS)})
    return runs


def _curve(path: Path) -> list[list[float]]:
    if not path.is_file():
        return []
    with np.load(path) as prediction:
        precision, recall, _ = precision_recall_curve(prediction["labels"], prediction["scores"])
    count = min(140, len(precision))
    indices = np.unique(np.linspace(0, len(precision) - 1, count, dtype=int))
    return [[round(float(recall[i]), 5), round(float(precision[i]), 5)] for i in indices]


def run_detail(run_id: str, dataset: str) -> dict:
    if not RUN_PATTERN.fullmatch(run_id) or dataset not in DATASETS:
        raise ValueError("Corrida o dataset invalido")
    directory = OUTPUTS / run_id
    if not directory.is_dir():
        raise FileNotFoundError(run_id)
    kind = _run_kind(run_id)
    protocol = None
    if kind in ("comparison", "thesis"):
        rows = [row for row in _json(directory / "results.json") if row["dataset"] == dataset]
        if not rows:
            raise FileNotFoundError(f"{run_id}/{dataset}")
        candidates = []
        seed = min(row["seed"] for row in rows)
        prediction_dir = directory / dataset / f"seed_{seed}" / "predictions"
        if kind == "thesis":
            protocol = _json(directory / "protocol.json")
            curves = {name: _curve(directory / dataset / f"seed_{seed}" / name / "predictions.npz") for name in MODEL_FILE}
        else:
            curves = {name: _curve(prediction_dir / f"{filename}.npz") for name, filename in MODEL_FILE.items()}
    elif kind in ("hybrid", "minibatch"):
        if kind == "minibatch":
            rows = [row for row in _json(directory / "results.json") if row["dataset"] == dataset]
            protocol = _json(directory / "protocol.json")
            if not rows:
                raise FileNotFoundError(f"{run_id}/{dataset}")
        else:
            rows = _json(directory / dataset / "selected_test.json")
        candidates_file = directory / dataset / "candidate_validation.json"
        candidates = _json(candidates_file) if candidates_file.is_file() else []
        seed = min(row["seed"] for row in rows)
        prediction_dir = directory / dataset / "selected" / f"seed_{seed}" / "predictions"
        curves = {model: _curve(prediction_dir / f"{model}.npz")
                  for model in ("hybrid", "baseline") if kind == "minibatch" or model == "hybrid"}
    else:
        protocol = _json(directory / "protocol.json")
        if not protocol.get("test_evaluated") or protocol.get("dataset") != dataset:
            raise FileNotFoundError(f"{run_id}/{dataset}")
        rows = [{**row, "model": "hybrid"} for row in _json(_feature_results_file(directory)) if "metrics" in row]
        if not rows:
            raise FileNotFoundError(f"{run_id}/{dataset}")
        candidates = []
        seed = min(row["seed"] for row in rows)
        prediction_dir = directory / dataset / f"seed_{seed}" / "predictions"
        curves = {"hybrid": _curve(prediction_dir / "hybrid.npz")}
    summary_file = directory / dataset / "prepared" / "metrics" / "prepare_summary.json"
    summary = _json(summary_file) if summary_file.is_file() else None
    return {"id": run_id, "kind": kind, "dataset": dataset, "rows": rows,
            "candidates": candidates, "summary": summary, "curves": curves,
            "curve_seed": seed,
            "caveat": (protocol["caveat"] if kind == "thesis" else "Exploratorio: el test historico ya se habia consultado antes de esta busqueda."
                       if kind != "comparison" else "Tres semillas comparten un mismo periodo de test."),
            "feature_count": protocol.get("top_k") if kind == "feature_selection" else summary.get("model_feature_count") if summary else None,
            "feature_set": protocol.get("feature_set") if kind in ("minibatch", "thesis") else None,
            "shap": (_json(directory / dataset / "explanations" / "summary.json")
                     if kind == "thesis" and (directory / dataset / "explanations" / "summary.json").is_file() else None)}


class DashboardHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload) -> None:
        self._send(status, json.dumps(_finite_json(payload), ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/overview":
                return self._send_json(200, {"runs": list_runs(),
                                             "graphs": {dataset: (CACHE_DIR / f"{dataset}.json").is_file() for dataset in DATASETS}})
            if path.startswith("/api/graph/"):
                dataset = path.removeprefix("/api/graph/")
                if dataset not in DATASETS:
                    raise ValueError("Dataset invalido")
                return self._send_json(200, _json(CACHE_DIR / f"{dataset}.json"))
            if path.startswith("/api/run/"):
                parts = path.removeprefix("/api/run/").split("/")
                if len(parts) != 2:
                    raise ValueError("Ruta de corrida invalida")
                return self._send_json(200, run_detail(parts[0], parts[1]))
            if path.startswith("/api/shap/"):
                parts = path.removeprefix("/api/shap/").split("/")
                allowed = {"global_importance.png", "local_TP.png", "local_FP.png", "local_FN.png", "local_TN.png", "precision_recall.png"}
                if len(parts) != 3 or not RUN_PATTERN.fullmatch(parts[0]) or not parts[0].startswith("thesis_v1_") or parts[1] not in DATASETS or parts[2] not in allowed:
                    raise ValueError("Imagen SHAP invalida")
                image_file = OUTPUTS / parts[0] / parts[1] / "explanations" / parts[2]
                return self._send(200, image_file.read_bytes(), "image/png")
            asset = "index.html" if path in ("/", "") else path.lstrip("/")
            if asset not in {"index.html", "studio.css", "app.js", "model_viz.js"}:
                raise FileNotFoundError(asset)
            file_path = ASSETS / asset
            body = file_path.read_bytes()
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            return self._send(200, body, content_type + ("; charset=utf-8" if content_type.startswith("text/") or content_type == "application/javascript" else ""))
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return self._send_json(404, {"error": str(exc)})
        except Exception as exc:
            print(f"Dashboard error: {exc}", flush=True)
            return self._send_json(500, {"error": "Error interno al leer los resultados"})

    def log_message(self, format: str, *args) -> None:
        print("dashboard: " + format % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-prepare", action="store_true", help="Inicia solo el panel de resultados")
    parser.add_argument("--rebuild-cache", action="store_true", help="Reconstruye ejemplos de grafos desde los CSV")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 1 y 65535")
    if not args.no_prepare:
        for dataset in DATASETS:
            if args.rebuild_cache or not (CACHE_DIR / f"{dataset}.json").is_file():
                try:
                    build_dataset_cache(dataset)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"No se pudo preparar el grafo de {dataset}: {exc}. Los resultados siguen disponibles.", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Panel listo: http://127.0.0.1:{args.port}  (Ctrl+C para detener)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
