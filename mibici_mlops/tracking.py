"""Versionado de datasets con MLflow: cada version es una corrida con su huella, esquema y reporte."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import pandas as pd

from mibici_mlops.config import EXPERIMENT_DATA, MIBICI_BASE_URL, MLFLOW_TRACKING_URI


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DatasetVersioner:
    """Registra versiones de datos en un experimento de MLflow.

    Una version queda identificada por la etiqueta `version` (v0, v1, ...) y por la huella
    SHA-256 de su archivo. Si ya existe una corrida con la misma huella no se duplica.
    """

    def __init__(self, experiment: str = EXPERIMENT_DATA, tracking_uri: str = MLFLOW_TRACKING_URI):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment = mlflow.set_experiment(experiment)

    def _already_logged(self, sha: str) -> bool:
        runs = mlflow.search_runs(
            [self.experiment.experiment_id], filter_string=f"tags.sha256 = '{sha}'"
        )
        return not runs.empty

    def log_version(
        self,
        version: str,
        df: pd.DataFrame,
        sha: str,
        source: str,
        description: str,
        params: dict | None = None,
        metrics: dict | None = None,
        tables: dict[str, pd.DataFrame] | None = None,
        file: Path | None = None,
    ) -> str | None:
        if self._already_logged(sha):
            logger.info(f"{version} con huella {sha[:12]} ya estaba registrada")
            return None
        with mlflow.start_run(run_name=version) as run:
            mlflow.set_tags({"version": version, "sha256": sha, "descripcion": description})
            mlflow.log_params({"filas": len(df), "columnas": df.shape[1], **(params or {})})
            if metrics:
                mlflow.log_metrics(metrics)
            dataset = mlflow.data.from_pandas(df.head(10_000), source=source, name=version)
            mlflow.log_input(dataset, context=version)
            schema = pd.DataFrame(
                {
                    "columna": df.columns,
                    "tipo": df.dtypes.astype(str).values,
                    "nulos": df.isna().sum().values,
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                for name, table in {"esquema": schema, **(tables or {})}.items():
                    out = Path(tmp) / f"{name}.csv"
                    table.to_csv(out, index=False)
                    mlflow.log_artifact(str(out), artifact_path="datos")
            if file is not None:
                mlflow.log_artifact(str(file), artifact_path="datos")
            logger.success(f"{version} registrada en MLflow (run {run.info.run_id})")
            return run.info.run_id

    def log_raw(self, raw: pd.DataFrame, files: list) -> str | None:
        files_table = pd.DataFrame([asdict(f) for f in files])
        sha = hashlib.sha256("".join(files_table["sha256"]).encode()).hexdigest()
        latin = int((files_table["encoding"] == "latin-1").sum())
        return self.log_version(
            "v0_crudo",
            raw,
            sha,
            source=f"{MIBICI_BASE_URL}/en/open-data/",
            description="Union de los 12 CSV mensuales de 2025 sin modificar valores",
            params={"archivos": len(files), "archivos_latin1": latin},
            tables={"archivos_crudos": files_table},
        )

    def log_clean(self, clean: pd.DataFrame, path: Path, rules, report: list[dict]) -> str | None:
        report_df = pd.DataFrame(report)
        raw_rows = int(report_df["filas_tras"].iloc[0])
        removed = raw_rows - len(clean)
        variant = path.stem.removeprefix("viajes_limpios_")
        return self.log_version(
            f"v1_limpio_{variant}",
            clean,
            file_sha256(path),
            source=str(path),
            description=f"Viajes limpios, variante {variant}",
            params={"variante": variant, **rules.as_params()},
            metrics={"filas_eliminadas": removed, "pct_eliminado": 100 * removed / raw_rows},
            tables={"reporte_limpieza": report_df},
        )
