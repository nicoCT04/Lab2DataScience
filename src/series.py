from pathlib import Path
import numpy as np
import pandas as pd

RUTAS_DATOS = [
    Path("data/raw/Base_Migracion_2009-2026jun.xlsx"),
    Path("../data/raw/Base_Migracion_2009-2026jun.xlsx"),
]

PAISES = ["El Salvador", "Estados Unidos de América", "Honduras"]

# Metadatos por serie: categoría (para el inciso 2.10) y transformación de varianza usada en Lab 1.
META = {
    "Total":          dict(categoria="Total", transform="log"),
    "Aérea":          dict(categoria="Vía",   transform="log"),
    "Terrestre":      dict(categoria="Vía",   transform="log"),
    "Marítima":       dict(categoria="Vía",   transform="log1p"),  # cruceristas, con ceros
    "El Salvador":    dict(categoria="País",  transform="log1p"),  # ceros en cierre pandémico
    "Estados Unidos": dict(categoria="País",  transform="log1p"),
    "Honduras":       dict(categoria="País",  transform="log1p"),
}


def cargar_datos(ruta=None) -> pd.DataFrame:
    """Lee la hoja `Datos`, construye el índice temporal mensual y devuelve el DataFrame."""
    if ruta is None:
        ruta = next((r for r in RUTAS_DATOS if r.exists()), None)
    if ruta is None or not Path(ruta).exists():
        raise FileNotFoundError("No se encontró data/raw/Base_Migracion_2009-2026jun.xlsx")
    df = pd.read_excel(ruta, sheet_name="Datos")
    df["Fecha"] = pd.to_datetime(dict(year=df["Año"], month=df["Mes cod"], day=1))
    return df


def construir_series(df: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """Devuelve un dict con las 7 series mensuales, idénticas a las del Laboratorio 1."""
    if df is None:
        df = cargar_datos()

    # Filtro base consistente en todo el período (definición estable del Lab 1).
    base = df[df["Tipo de Viajero"].isin(["Turista", "Excursionista"])]
    rango = pd.date_range(df["Fecha"].min(), df["Fecha"].max(), freq="MS")

    series: dict[str, pd.Series] = {}

    # Total y vías Aérea / Terrestre (Turista + Excursionista) 
    series["Total"] = base.groupby("Fecha")["Viajero"].sum().sort_index().asfreq("MS")
    for via in ["Aérea", "Terrestre"]:
        s = base[base["Vía"] == via].groupby("Fecha")["Viajero"].sum().sort_index().asfreq("MS")
        series[via] = s

    #Marítima: Cruceristas, ventana 2009-2022 (ceros = temporada baja / cierre)
    rango_mar = pd.date_range("2009-01-01", "2022-12-01", freq="MS")
    mar = (df[(df["Vía"] == "Marítima") & (df["Tipo de Viajero"] == "Cruceristas")]
           .groupby("Fecha")["Viajero"].sum().reindex(rango_mar).fillna(0.0).asfreq("MS"))
    series["Marítima"] = mar

    #  Países (El Salvador, EE.UU., Honduras): ceros en abr-ago 2020 (cierre pandémico) 
    claves = {"El Salvador": "El Salvador",
            "Estados Unidos": "Estados Unidos de América",
            "Honduras": "Honduras"}
    for clave, pais in claves.items():
        s = base[base["País"] == pais].groupby("Fecha")["Viajero"].sum().reindex(rango).fillna(0.0).asfreq("MS")
        series[clave] = s

    for nombre, s in series.items():
        s.name = nombre
    return series


if __name__ == "__main__":
    series = construir_series()
    print(f"{'serie':16s} {'n':>4s}  {'inicio':>10s}  {'fin':>10s}  {'ceros':>5s}  categoría")
    for nombre, s in series.items():
        print(f"{nombre:16s} {len(s):4d}  {str(s.index.min().date()):>10s}  "
              f"{str(s.index.max().date()):>10s}  {int((s == 0).sum()):5d}  {META[nombre]['categoria']}")
