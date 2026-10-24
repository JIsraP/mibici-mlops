import mlflow
import pandas as pd
import pytest

from mibici_mlops.tracking import DatasetVersioner


@pytest.fixture
def versioner(tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    tracker = DatasetVersioner(
        experiment="prueba_versionado",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    yield tracker
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)


def log_sample(versioner, version="v1_prueba", sha="huella_a"):
    return versioner.log_version(
        version,
        pd.DataFrame({"salidas": [1, 2]}),
        sha,
        source="prueba.csv",
        description="Datos de prueba",
    )


def test_repetir_version_reutiliza_corrida_terminada(versioner):
    first_id = log_sample(versioner)
    second_id = log_sample(versioner)

    runs = mlflow.search_runs([versioner.experiment.experiment_id])
    assert second_id == first_id
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "FINISHED"
    assert runs.iloc[0]["tags.estado"] == "vigente"


def test_misma_huella_con_version_distinta_se_registra(versioner):
    first_id = log_sample(versioner, version="v1_prueba")
    second_id = log_sample(versioner, version="v2_prueba")

    runs = mlflow.search_runs([versioner.experiment.experiment_id])
    assert first_id != second_id
    assert set(runs["tags.version"]) == {"v1_prueba", "v2_prueba"}


def test_corrida_fallida_permite_reintentar(versioner, monkeypatch):
    def fail_upload(*args, **kwargs):
        raise RuntimeError("Fallo de prueba al guardar el artefacto")

    with monkeypatch.context() as patch:
        patch.setattr(mlflow, "log_artifact", fail_upload)
        with pytest.raises(RuntimeError, match="Fallo de prueba"):
            log_sample(versioner)

    run_id = log_sample(versioner)
    runs = mlflow.search_runs([versioner.experiment.experiment_id])
    assert len(runs) == 2
    assert set(runs["status"]) == {"FAILED", "FINISHED"}
    assert mlflow.get_run(run_id).info.status == "FINISHED"


def test_corrida_descartada_no_impide_nuevo_registro(versioner):
    discarded_id = log_sample(versioner)
    mlflow.MlflowClient().set_tag(discarded_id, "estado", "descartada")

    current_id = log_sample(versioner)
    assert current_id != discarded_id
    assert mlflow.get_run(current_id).data.tags["estado"] == "vigente"
    assert mlflow.get_run(discarded_id).data.tags["estado"] == "descartada"


def test_cambio_de_datos_crea_otra_corrida(versioner):
    first_id = log_sample(versioner, sha="huella_a")
    second_id = log_sample(versioner, sha="huella_b")

    runs = mlflow.search_runs([versioner.experiment.experiment_id])
    assert first_id != second_id
    assert len(runs) == 2


def test_corrida_anterior_sin_estado_se_puede_reutilizar(versioner):
    previous_id = log_sample(versioner)
    mlflow.MlflowClient().delete_tag(previous_id, "estado")
    with mlflow.start_run(run_name="version_descartada"):
        mlflow.set_tags({"version": "v1_prueba", "sha256": "huella_a", "estado": "descartada"})

    reused_id = log_sample(versioner)
    runs = mlflow.search_runs([versioner.experiment.experiment_id])
    assert reused_id == previous_id
    assert len(runs) == 2
