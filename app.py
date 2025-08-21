from typing import List, Dict, Any, Optional from fastapi import FastAPI from fastapi.middleware.cors import CORSMiddleware from pydantic import BaseModel import pandas as pd import requests from io import StringIO

URLs de tus hojas (se pueden cambiar luego por variables de entorno)
URL_PORTFOLIO = "https://docs.google.com/spreadsheets/d/1VoQ1Y7iw8V0DCLGe9cUnvPjVdG-IxmzNdyls7Bj2w8I/export?format=csv" URL_COMPRAS = "https://docs.google.com/spreadsheets/d/1erlpqJOqiNBe0UikJD1T-h1aTu7beaJd/export?format=csv" URL_OCUP2024 = "https://docs.google.com/spreadsheets/d/1XVNeaqFWFOt_g2TVY1YxyE3bmxvVEkYUK8PJQOR4kvc/export?format=csv"

Carga CSV desde Google Sheets
def fetch_csv(url: str, decimal: Optional[str]=None) -> pd.DataFrame: r = requests.get(url, timeout=30) r.raise_for_status() return pd.read_csv(StringIO(r.text), encoding="utf-8", decimal=decimal)

app = FastAPI(title="Hotels RAG Demo") app.add_middleware( CORSMiddleware, allow_origins=[""], # para demo, abierto allow_credentials=False, allow_methods=[""], allow_headers=["*"], )

portfolio_df: Optional[pd.DataFrame] = None compras_df: Optional[pd.DataFrame] = None ocup_df: Optional[pd.DataFrame] = None

def load_data(): global portfolio_df, compras_df, ocup_df portfolio_df = fetch_csv(URL_PORTFOLIO) compras_df = fetch_csv(URL_COMPRAS, decimal=",") ocup_df = fetch_csv(URL_OCUP2024)

# Limpieza básica
if "País" in portfolio_df.columns:
    portfolio_df["País"] = portfolio_df["País"].astype(str).str.strip()
if "Habitaciones" in portfolio_df.columns:
    portfolio_df["Habitaciones"] = pd.to_numeric(portfolio_df["Habitaciones"], errors="coerce").fillna(0).astype(int)
for col in ("Marca","Propiedad","Ciudad","Tipo"):
    if col in portfolio_df.columns:
        portfolio_df[col] = portfolio_df[col].astype(str).str.strip()

if "fecha" in compras_df.columns:
    compras_df["fecha"] = pd.to_datetime(compras_df["fecha"], errors="coerce")
for c in ("unidades","precio_unitario","precio_total"):
    if c in compras_df.columns:
        compras_df[c] = pd.to_numeric(compras_df[c], errors="coerce")
for c in ("hotel","proveedor"):
    if c in compras_df.columns:
        compras_df[c] = compras_df[c].astype(str).str.strip()

for c in ["YTD","Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]:
    if c in ocup_df.columns:
        ocup_df[c] = pd.to_numeric(ocup_df[c], errors="coerce")
for col in ("Marca","Propiedad","Ciudad","País","Tipo"):
    if col in ocup_df.columns:
        ocup_df[col] = ocup_df[col].astype(str).str.strip()
@app.on_event("startup") def on_startup(): load_data()

@app.get("/health") def health(): return {"status": "ok"}

@app.post("/reload") def reload_data(): load_data() return {"status":"reloaded"}

Modelos de entrada
class Filters(BaseModel): country: Optional[str] = None city: Optional[str] = None brand: Optional[str] = None property: Optional[str] = None type: Optional[str] = None year: Optional[int] = None month: Optional[str] = None domain: Optional[str] = "all"

class Aggregate(BaseModel): metric: Optional[str] = None group_by: Optional[List[str]] = None weighting: Optional[str] = None

class QueryRequest(BaseModel): query: str filters: Optional[Filters] = None top_k: Optional[int] = 8 rerank: Optional[bool] = True aggregate: Optional[Aggregate] = None return_sources: Optional[bool] = True

def source_list(): return [ {"title":"Portfolio", "url":URL_PORTFOLIO, "note":"Marca, Propiedad, Ciudad, País, Tipo, Habitaciones"}, {"title":"Compras", "url":URL_COMPRAS, "note":"compras, unidades, referencia_producto, precio_unitario, precio_total, hotel, fecha, proveedor"}, {"title":"Ocupación 2024", "url":URL_OCUP2024, "note":"Marca, Propiedad, Ciudad, País, Tipo, YTD, Ene...Dic"}, ]

@app.post("/query") def query(req: QueryRequest): f = req.filters or Filters() agg = req.aggregate or Aggregate()

# 1) Habitaciones por país (portfolio)
if agg.metric == "rooms_total" and (agg.group_by or []) == ["country"]:
    df = portfolio_df.copy()
    if f.country:
        df = df[df["País"].str.lower() == f.country.lower()]
    out = (df.groupby("País", dropna=False)["Habitaciones"]
             .sum()
             .reset_index()
             .rename(columns={"País":"country","Habitaciones":"rooms_total"})
             .sort_values("rooms_total", ascending=False)
             .to_dict(orient="records"))
    return {"answer":"Total de habitaciones por país (portfolio).",
            "aggregates": out, "sources": [s for s in source_list() if s["title"]=="Portfolio"]}

# 2) YTD 2024 por propiedad en ciudad (ocupación)
if f.city and f.year == 2024 and (agg.group_by or []) == ["property"]:
    df = ocup_df.copy()
    df = df[df["Ciudad"].str.lower() == f.city.lower()]
    out = (df[["Propiedad","YTD"]]
             .rename(columns={"Propiedad":"property"})
             .sort_values("YTD", ascending=False)
             .to_dict(orient="records"))
    return {"answer": f"Ranking YTD 2024 de ocupación en {f.city}.",
            "aggregates": out, "sources": [s for s in source_list() if s["title"]=="Ocupación 2024"]}

# 3) Coste total compras 2023 por hotel y proveedor
if f.year == 2023 and set(agg.group_by or []) == {"property","proveedor"} and agg.metric == "importe_total":
    df = compras_df.copy()
    df = df[df["fecha"].dt.year == 2023]
    out = (df.groupby(["hotel","proveedor"], dropna=False)["precio_total"]
             .sum()
             .reset_index()
             .rename(columns={"hotel":"property","precio_total":"importe_total"})
             .sort_values("importe_total", ascending=False)
             .to_dict(orient="records"))
    return {"answer":"Coste total de compras 2023 por hotel y proveedor.",
            "aggregates": out, "sources": [s for s in source_list() if s["title"]=="Compras"]}

# Fallback
return {"answer":"Demo: prueba 1) rooms_total por country 2) YTD 2024 por property en city 3) importe_total 2023 por property y proveedor.",
        "aggregates":[], "sources": source_list()}
