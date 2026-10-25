import pandas as pd
import pytest

from mibici_mlops.features import build_demand
from mibici_mlops.model_input import load_official_trips, model_trips
from mibici_mlops.validation import validate_demand

HEADER = (
    "Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,"
    "Origen_Id,Destino_Id,duracion_segundos,duracion_no_positiva,duracion_mayor_24h"
)


@pytest.fixture
def official_dir(tmp_path):
    rows = [
        "1,10,M,1990.0,2025-03-01 08:00:00,2025-03-01 08:12:00,1,2,720.0,False,False",
        "2,11,,1985.0,2025-03-01 08:00:00,2025-03-01 08:00:00,1,2,0.0,True,False",
        "3,12,F,,2025-03-01 09:00:00,2025-03-02 12:00:00,2,1,97200.0,False,True",
        "4,13,F,2000.0,01/03/2025 10:00,01/03/2025 10:20,2,2,1200.0,False,False",
        "5,14,M,1995.0,2025-12-31 23:50:00,2026-01-01 00:10:00,1,2,1200.0,False,False",
    ]
    (tmp_path / "datos_abiertos_2025_03.csv").write_text(
        HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_lee_banderas_y_ambos_formatos_de_fecha(official_dir):
    trips, files = load_official_trips(official_dir)
    assert len(files) == 1 and len(trips) == 5
    assert trips["duracion_no_positiva"].tolist() == [False, True, False, False, False]
    assert trips.loc[3, "inicio"] == pd.Timestamp("2025-03-01 10:00")
    assert trips.loc[1, "genero"] == "Desconocido"


def test_modelo_excluye_solo_duracion_no_positiva(official_dir):
    trips, _ = load_official_trips(official_dir)
    model, counts = model_trips(trips)
    assert sorted(model["viaje_id"]) == [1, 3, 4, 5]  # el de mas de 24 h se conserva
    assert counts == {
        "marcados_duracion_no_positiva": 1,
        "marcados_duracion_mayor_24h": 1,
        "excluidos_para_modelo": 1,
    }


def test_contrato_incompleto_falla(tmp_path):
    (tmp_path / "datos_abiertos_2025_01.csv").write_text("Viaje_Id,Usuario_Id\n1,2\n")
    with pytest.raises(ValueError, match="contrato"):
        load_official_trips(tmp_path)


def test_validacion_de_demanda(official_dir):
    model, _ = model_trips(load_official_trips(official_dir)[0])
    demand = build_demand(model)
    report = {r["prueba"]: r for r in validate_demand(demand, model, {1, 2})}
    assert all(r["resultado"] == "cumple" for r in report.values())
    assert "1 terminan en 2026" in report["llegadas cuadran con viajes"]["detalle"]


def test_validacion_detecta_horas_faltantes(official_dir):
    model, _ = model_trips(load_official_trips(official_dir)[0])
    demand = build_demand(model).iloc[1:]
    report = {r["prueba"]: r for r in validate_demand(demand, model, {1, 2})}
    assert report["horas completas por estacion"]["resultado"] == "no cumple"
    assert report["salidas cuadran con viajes"]["resultado"] == "cumple"
