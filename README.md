# Laboratorio 2 — Deep Learning / Series de Tiempo (CC3084 Data Science, UVG, Sem. II 2026)

Modelos **LSTM** para predecir las series de tiempo del **ingreso de viajeros internacionales a
Guatemala** (ene-2009 a jun-2026) y comparación con los modelos clásicos del Laboratorio 1.
Además, exploración de la similitud entre series con el algoritmo **catch22**. Datos de uso
exclusivamente académico.

## Estado de entrega

- [x] **Avance (pasaporte)** — Ejercicio 1: ≥2 modelos LSTM por serie con tuneo de hiperparámetros,
      selección del mejor, predicción y comparación con el Lab 1. → `notebooks/Lab2_LSTM.ipynb`
- [~] **Ejercicio 2.14** — nuevo LSTM con características de catch22 (serie Aérea). → `notebooks/Lab2_LSTM.ipynb`
- [ ] **Ejercicio 2.1–2.13** — extracción catch22 para las 7 series, matriz, PCA, clustering,
      heatmaps, matriz de distancias e interpretación. → `notebooks/Lab2_catch22.ipynb`

## Contenido

```text
.
├── data/raw/Base_Migracion_2009-2026jun.xlsx   # misma base del Laboratorio 1
├── src/
│   └── series.py                               # construcción de las 7 series (compartido)
├── notebooks/
│   ├── Lab2_LSTM.ipynb                          # Ej.1 (LSTM Total y Aérea) + Ej.2.14 (LSTM con catch22)
│   └── Lab2_catch22.ipynb                       # Ej.2.1–2.13 — exploración catch22  [pendiente]
├── Laboratorio 2. Deep Learning_Series.pdf      # enunciado
└── README.md
```

## Reparto del trabajo (2 personas)

Trabajo grupal dividido en dos notebooks para avanzar en paralelo; ambos reutilizan `src/series.py`
para construir las series de forma idéntica al Lab 1:

- **Track modelado** (`Lab2_LSTM.ipynb`): Ejercicio 1 + inciso 2.14 (LSTM con catch22).
- **Track exploración** (`Lab2_catch22.ipynb`): incisos 2.1–2.13 sobre las 7 series.

## Series y partición

Se usan **las mismas definiciones y el mismo split 70/30** del Laboratorio 1:

| Serie | Definición | Transformación | train / test |
|-------|-----------|----------------|--------------|
| **Total** | `Turista + Excursionista` (mensual) | `log` | 147 / 63 |
| **Aérea** | Vía de ingreso *Aérea* (`Turista + Excursionista`) | `log` | 147 / 63 |

`train` = ene-2009 a mar-2021 · `test` = abr-2021 a jun-2026 (recuperación post-pandemia).

## Cómo ejecutar

> **Importante:** TensorFlow aún no publica *wheels* para Python 3.14, así que el entorno usa
> **Python 3.13**.

```bash
python3.13 -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn statsmodels openpyxl ipykernel pycatch22
python -m ipykernel install --user --name lab2-ds --display-name "Python 3.13 (Lab2 DS)"
```

Abrir `notebooks/Lab2_LSTM.ipynb`, seleccionar el kernel **Python 3.13 (Lab2 DS)** y ejecutar de
arriba hacia abajo. El archivo de datos debe conservarse en
`data/raw/Base_Migracion_2009-2026jun.xlsx`.

## Metodología del Ejercicio 1

Los LSTM aprenden las **log-diferencias** de cada serie (la misma `d=1` que usan los ARIMA del
Lab 1), lo que hace la comparación *apples-to-apples* y estabiliza el pronóstico recursivo. Por serie
se evalúan 6 configuraciones (rejilla sobre `look_back`, unidades, capas y dropout), se selecciona la
mejor por pérdida de validación interna (sin mirar el test) y se produce un **pronóstico recursivo**
de los 63 meses de prueba, comparándolo con el mejor modelo del Laboratorio 1 (ARIMA(1,1,1)).
