import pandas as pd
import pytest

from mibici.cleaning import CleaningRules, TripCleaner
from mibici.dataset import detect_encoding
from mibici.features import build_demand


def raw_trip(viaje, inicio, fin, origen="1", destino="2", genero="M", anio="1990"):
    return [str(viaje), "10", genero, anio, inicio, fin, origen, destino]


@pytest.fixture
def raw():
    rows = [
        raw_trip(1, "2025-03-01 08:00:00", "2025-03-01 08:12:00"),  # valido
        raw_trip(2, "2025-03-01 08:00:00", "2025-03-01 08:00:00"),  # duracion cero
        raw_trip(3, "2025-03-01 08:00:00", "2025-03-01 08:01:00", "1", "1"),  # arranque fallido
        raw_trip(4, "2025-03-01 08:00:00", "2025-03-01 08:00:30"),  # muy corto
        raw_trip(5, "2025-03-01 08:00:00", "2025-03-01 12:00:00"),  # atipico
        raw_trip(6, "2025-03-01 09:00:00", "2025-03-01 09:20:00", "1", "9"),  # fuera de catalogo
        raw_trip(7, "2025-03-01 09:00:00", "2025-03-01 09:10:00", genero="NULL", anio="199"),
        raw_trip(1, "2025-03-01 08:00:00", "2025-03-01 08:12:00"),  # duplicado
    ]
    cols = ["viaje_id", "usuario_id", "genero", "anio_nacimiento", "inicio", "fin"]
    return pd.DataFrame(rows, columns=cols + ["origen_id", "destino_id"])


def test_estricta_quita_lo_invalido(raw):
    cleaner = TripCleaner(CleaningRules.variant("estricta"), valid_station_ids={1, 2})
    clean = cleaner.clean(raw)
    assert sorted(clean["viaje_id"]) == [1, 7]
    report = cleaner.report_frame().set_index("regla")["filas_afectadas"]
    assert report["viaje_id_duplicado"] == 1
    assert report["arranque_fallido"] == 1
    assert report["duracion_atipica"] == 1


def test_genero_y_anio_se_imputan_sin_borrar_el_viaje(raw):
    clean = TripCleaner(CleaningRules(), {1, 2}).clean(raw).set_index("viaje_id")
    assert clean.loc[7, "genero"] == "Desconocido"
    assert pd.isna(clean.loc[7, "anio_nacimiento"])


def test_laxa_conserva_viajes_cortos_y_largos(raw):
    clean = TripCleaner(CleaningRules.variant("laxa"), {1, 2}).clean(raw)
    assert sorted(clean["viaje_id"]) == [1, 3, 4, 5, 7]


def test_variante_desconocida():
    with pytest.raises(ValueError):
        CleaningRules.variant("otra")


def test_detecta_latin1(tmp_path):
    f = tmp_path / "mes.csv"
    f.write_bytes("A\xf1o\n1".encode("latin-1"))
    assert detect_encoding(f) == "latin-1"


def test_demanda_rellena_horas_vacias(raw):
    clean = TripCleaner(CleaningRules(), {1, 2}).clean(raw)
    demand = build_demand(clean)
    assert len(demand) == 2 * 8760  # dos estaciones por todas las horas de 2025
    assert demand["salidas"].sum() == len(clean) == demand["llegadas"].sum()
    hit = demand[(demand["estacion_id"] == 1) & (demand["fecha_hora"] == "2025-03-01 08:00")]
    assert hit["salidas"].item() == 1
