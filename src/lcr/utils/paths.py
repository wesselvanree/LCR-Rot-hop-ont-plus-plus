from pathlib import Path


class Paths:
    repo_root = Path(__file__).parent.parent.parent.parent

    data_raw = repo_root / "data" / "raw"
    data_processed = repo_root / "data" / "processed"

    models = repo_root / "output" / "models"
    embeddings = repo_root / "output" / "embeddings"
