import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("Non-JSON YAML requires PyYAML. Install requirements.txt.") from error
        return yaml.safe_load(text)

