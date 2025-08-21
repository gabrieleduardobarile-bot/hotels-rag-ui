# app.py
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests
from io import StringIO
import os
import re

# URLs de datos (Google Sheets export=csv)
URL_PORTFOLIO = "https://docs.google.com/spreadsheets/d/1VoQ1Y7iw8V0DCLGe9cUnvPjVdG-IxmzNdyls7Bj2w8I/export?format=csv"
URL_COMPRAS   = "https://docs.google.com/spreadsheets/d/1erlpqJOqiNBe0UikJD1T-h1aTu7beaJd/export?format=csv"
URL_OCUP2024  = "https://docs.google.com/spreadsheets/d/1XVNeaqFWFOt_g2TVY1YxyE3bmxvVEkYUK8PJQOR4kvc/export?format=csv"

app = FastAPI(title="Hotels RAG Demo (NL→JSON)")

# CORS abierto (útil si accedes desde otros orígenes; con Nginx proxy no sería estrictamente necesario)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DataFrames globales
DF_PORT: Optional[pd.DataFrame] = None
DF_COMP: Optional[pd.DataFrame] = None
DF_OCUP: Optional[pd.DataFrame] = None


def download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def load_data():
    global DF_PORT, DF_COMP, DF_OCUP
    DF_PORT = download_csv(URL_PORTFOLIO)
    DF_COMP = download_csv(URL_COMPRAS)
    DF_OCUP = download_csv(URL_OCUP2024)


def ensure_loaded():
    global DF_PORT, DF_COMP, DF_OCUP
    if DF_PORT is None or DF_COMP is None or DF_OCUP is None:
        load_data()


def source_list():
    return [
        {"title": "Portfolio", "url": URL_PORTFOLIO, "note": "Marca, Propiedad, Ciudad, País, Tipo, Habitaciones"},
        {"title": "Compras", "url": URL_COMPRAS, "note": "Compras 2023–2025 (dummy) hotel/proveedor"},
        {"title": "Ocupación 2024", "url": URL_OCUP2024, "note": "YTD y Ene–Dic por propiedad/ciudad"},
    ]


# Modelos de entrada
class Filters(BaseModel):
    country: Optional[str] = None
    city: Optional[str] = None
    brand: Optional[str] = None
    property: Optional[str] = None
    type: Optional[str] = None
    year: Optional[int] = None
    month: Optional[str] = None
    domain: Optional[str] = "all"


class Aggregate(BaseModel):
    metric: Optional[str] = None
    group_by: Optional[list[str]] = None
    weighting: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    filters: Optional[Filters] = None
    top_k: Optional[int] = 8
    rerank: Optional[bool] = True
    aggregate: Optional[Aggregate] = None
    return_sources: Optional[bool] = True


# Intérprete NL → QueryRequest (heurística rápida, sin LLM)
def interpret_nl_to_queryrequest(q: str) -> dict:
    """
    Convierte texto libre a un payload que /query entiende (heurísticas mínimas):
      1) 'habitaciones' + 'país/pais/country' => rooms_total por country
      2) 'ocup' o 'ytd' + '2024' (+ opcional 'en <ciudad>') => group_by property; year=2024; city opcional
      3) 'compra(s)' + '2023' => importe_total 2023 por property y proveedor
    Si no matchea, devuelve solo {"query": q} y el endpoint mostrará guía/fuentes.
    """
    if not q:
        return {"query": ""}

    text = q.lower()

    # 1) Habitaciones por país
    if ("habitacion" in text or "habitaciones" in text) and ("país" in text or "pais" in text or "country" in text):
        return {
            "query": q,
            "aggregate": {"metric": "rooms_total", "group_by": ["country"]}
        }

    # 2) Ocupación YTD 2024 (opcional "en <ciudad>")
    if ("ocup" in text or "ytd" in text) and ("2024" in text):
        city = None
        m = re.search(r"\ben\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+){0,2})\b", q)
        if m:
            city = m.group(1).strip()
        payload = {"query": q, "aggregate": {"group_by": ["property"]}}
        payload["filters"] = {"year": 2024}
        if city:
            payload["filters"]["city"] = city
        return payload

    # 3) Compras 2023 por hotel y proveedor
    if ("compra" in text or "compras" in text) and "2023" in text:
        return {
            "query": q,
            "filters": {"year": 2023},
            "aggregate": {"metric": "importe_total", "group_by": ["property", "proveedor"]}
        }

    # Por defecto
    return {"query": q}


# Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reload")
def reload_data():
    load_data()
    return {"status": "reloaded"}


@app.post("/query")
def query(req: QueryRequest):
    ensure_loaded()

    # Auto-interpretación: si no llegan aggregate/filters, deducirlos desde el texto
    if (req.aggregate is None) and (req.filters is None) and (req.query or "").strip():
        interpreted = interpret_nl_to_queryrequest(req.query)
        if "aggregate" in interpreted and interpreted["aggregate"]:
            req.aggregate = Aggregate(**interpreted["aggregate"])
        if "filters" in interpreted and interpreted["filters"]:
            req.filters = Filters(**interpreted["filters"])

    f = req.filters or Filters()
    agg = req.aggregate or Aggregate()

    # Caso 1: total de habitaciones por país
    if agg.metric == "rooms_total" and (agg.group_by or []) == ["country"]:
        df = DF_PORT.copy()
        df = df.rename(columns={"País": "country", "Habitaciones": "rooms_total"})
        df["rooms_total"] = pd.to_numeric(df["rooms_total"], errors="coerce")
        out = (
            df.groupby("country", dropna=False)["rooms_total"]
            .sum(min_count=1)
            .reset_index()
            .sort_values("rooms_total", ascending=False)
        )
        return {
            "answer": "Total de habitaciones por país (portfolio).",
            "aggregates": out.to_dict(orient="records"),
            "sources": [s for s in source_list() if s["title"] == "Portfolio"],
        }

    # Caso 2: ranking YTD 2024 por propiedad en una ciudad
    if f.year == 2024 and (agg.group_by or []) == ["property"]:
        df = DF_OCUP.copy()
        df = df.rename(columns={"Propiedad": "property"})
        if f.city:
            df = df[df["Ciudad"].astype(str).str.lower() == f.city.lower()]
        # Asegurar numérico para YTD
        if "YTD" in df.columns:
            df["YTD"] = pd.to_numeric(df["YTD"], errors="coerce")
        out = (
            df[["property", "YTD"]]
            .sort_values("YTD", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        city_lbl = f" en {f.city}" if f.city else ""
        return {
            "answer": f"Ranking YTD 2024 de ocupación por propiedad{city_lbl}.",
            "aggregates": out.to_dict(orient="records"),
            "sources": [s for s in source_list() if s["title"] == "Ocupación 2024"],
        }

    # Caso 3: compras 2023 por hotel (property) y proveedor con importe_total
    if f.year == 2023 and agg.metric == "importe_total" and set(agg.group_by or []) == {"property", "proveedor"}:
        df = DF_COMP.copy()
        df = df.rename(columns={"hotel": "property", "precio_total": "importe_total"})
        # Filtrado por año si hay fecha
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df[df["fecha"].dt.year == 2023]
        df["importe_total"] = pd.to_numeric(df["importe_total"], errors="coerce")
        out = (
            df.groupby(["property", "proveedor"], dropna=False)["importe_total"]
            .sum(min_count=1)
            .reset_index()
            .sort_values(["importe_total"], ascending=False, na_position="last")
        )
        return {
            "answer": "Coste total de compras 2023 por hotel y proveedor.",
            "aggregates": out.to_dict(orient="records"),
            "sources": [s for s in source_list() if s["title"] == "Compras"],
        }

    # Por defecto, mostrar guía y fuentes
    return {
        "answer": "Consulta no reconocida. Prueba: 1) 'Total de habitaciones por país' 2) 'Ocupación YTD 2024 por propiedad en <ciudad>' 3) 'Coste total de compras 2023 por hotel y proveedor'.",
        "aggregates": [],
        "sources": source_list(),
    }


# Arranque seguro: no tumbar el servicio si falla la carga
@app.on_event("startup")
def on_startup():
    if os.getenv("SKIP_LOAD_ON_START") == "1":
        print("Skipping load_data on startup")
        return
    try:
        load_data()
        print("Data loaded on startup")
    except Exception as e:
        print("WARN: load_data on startup failed:", e)
