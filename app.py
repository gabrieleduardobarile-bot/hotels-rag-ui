# app.py
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests
from io import StringIO
import os

# URLs de tus hojas
URL_PORTFOLIO = "https://docs.google.com/spreadsheets/d/1VoQ1Y7iw8V0DCLGe9cUnvPjVdG-IxmzNdyls7Bj2w8I/export?format=csv"
URL_COMPRAS   = "https://docs.google.com/spreadsheets/d/1erlpqJOqiNBe0UikJD1T-h1aTu7beaJd/export?format=csv"
URL_OCUP2024  = "https://docs.google.com/spreadsheets/d/1XVNeaqFWFOt_g2TVY1YxyE3bmxvVEkYUK8PJQOR4kvc/export?format=csv"

app = FastAPI(title="Hotels RAG Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DF_PORT = None
DF_COMP = None
DF_OCUP = None

def download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def load_data():
    global DF_PORT, DF_COMP, DF_OCUP
    DF_PORT = download_csv(URL_PORTFOLIO)
    DF_COMP = download_csv(URL_COMPRAS)
    DF_OCUP = download_csv(URL_OCUP2024)

def source_list():
    return [
        {"title": "Portfolio", "url": URL_PORTFOLIO, "note": "Marca, Propiedad, Ciudad, País, Tipo, Habitaciones"},
        {"title": "Compras", "url": URL_COMPRAS, "note": "Compras 2023–2025 (dummy) hotel/proveedor"},
        {"title": "Ocupación 2024", "url": URL_OCUP2024, "note": "YTD y Ene–Dic por propiedad/ciudad"},
    ]

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reload")
def reload_data():
    load_data()
    return {"status": "reloaded"}

def ensure_loaded():
    global DF_PORT, DF_COMP, DF_OCUP
    if DF_PORT is None or DF_COMP is None or DF_OCUP is None:
        load_data()

@app.post("/query")
def query(req: QueryRequest):
    ensure_loaded()
    f = req.filters or Filters()
    agg = req.aggregate or Aggregate()

    # Caso 1: total de habitaciones por país
    if agg.metric == "rooms_total" and (agg.group_by or []) == ["country"]:
        df = DF_PORT.copy()
        df = df.rename(columns={"País": "country", "Habitaciones": "rooms_total"})
        # Asegurar numérico
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
    if f.city and f.year == 2024 and (agg.group_by or []) == ["property"]:
        df = DF_OCUP.copy()
        df = df.rename(columns={"Propiedad": "property"})
        df_city = df[df["Ciudad"].astype(str).str.lower() == f.city.lower()]
        # Asegurar numérico
        df_city["YTD"] = pd.to_numeric(df_city["YTD"], errors="coerce")
        out = (
            df_city[["property", "YTD"]]
            .sort_values("YTD", ascending=False)
            .reset_index(drop=True)
        )
        return {
            "answer": f"Ranking YTD 2024 de ocupación en {f.city}.",
            "aggregates": out.to_dict(orient="records"),
            "sources": [s for s in source_list() if s["title"] == "Ocupación 2024"],
        }

    # Caso 3: compras 2023 por hotel (property) y proveedor con importe_total
    if f.year == 2023 and set(agg.group_by or []) == {"property", "proveedor"} and agg.metric == "importe_total":
        df = DF_COMP.copy()
        df = df.rename(columns={"hotel": "property", "precio_total": "importe_total"})
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df = df[df["fecha"].dt.year == 2023]
        df["importe_total"] = pd.to_numeric(df["importe_total"], errors="coerce")
        out = (
            df.groupby(["property", "proveedor"], dropna=False)["importe_total"]
            .sum(min_count=1)
            .reset_index()
            .sort_values(["importe_total"], ascending=False)
        )
        return {
            "answer": "Coste total de compras 2023 por hotel y proveedor.",
            "aggregates": out.to_dict(orient="records"),
            "sources": [s for s in source_list() if s["title"] == "Compras"],
        }

    return {
        "answer": "Demo: prueba 1) rooms_total por country 2) YTD 2024 por property en city 3) importe_total 2023 por property y proveedor.",
        "aggregates": [],
        "sources": source_list(),
    }

@app.on_event("startup")
def on_startup():
    if os.getenv("SKIP_LOAD_ON_START") == "1":
        print("Skipping load_data on startup")
        return
    try:
        load_data()
    except Exception as e:
        print("WARN: load_data on startup failed:", e)
