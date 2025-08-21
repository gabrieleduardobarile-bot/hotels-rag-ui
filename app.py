# app.py
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests
from io import StringIO
import os
import re
import json

# ============ CONFIG ============

URL_PORTFOLIO = "https://docs.google.com/spreadsheets/d/1VoQ1Y7iw8V0DCLGe9cUnvPjVdG-IxmzNdyls7Bj2w8I/export?format=csv"
URL_COMPRAS   = "https://docs.google.com/spreadsheets/d/1erlpqJOqiNBe0UikJD1T-h1aTu7beaJd/export?format=csv"
URL_OCUP2024  = "https://docs.google.com/spreadsheets/d/1XVNeaqFWFOt_g2TVY1YxyE3bmxvVEkYUK8PJQOR4kvc/export?format=csv"

ENABLE_LLM = os.getenv("ENABLE_LLM", "0") == "1"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("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", "")

# ============ APP INIT ============

app = FastAPI(title="Hotels RAG Demo (NL→JSON + LLM opcional)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DF_PORT: Optional[pd.DataFrame] = None
DF_COMP: Optional[pd.DataFrame] = None
DF_OCUP: Optional[pd.DataFrame] = None

# ============ DATA LOADERS ============

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
    if any(df is None for df in [DF_PORT, DF_COMP, DF_OCUP]):
        load_data()

def source_list():
    return [
        {"title": "Portfolio", "url": URL_PORTFOLIO, "note": "Marca, Propiedad, Ciudad, País, Tipo, Habitaciones"},
        {"title": "Compras", "url": URL_COMPRAS, "note": "Compras 2023–2025 (dummy) hotel/proveedor, unidades, importes"},
        {"title": "Ocupación 2024", "url": URL_OCUP2024, "note": "YTD y meses Ene–Dic por propiedad/ciudad"},
    ]

# ============ SCHEMA ============

class Filters(BaseModel):
    country: Optional[str] = None
    city: Optional[str] = None
    brand: Optional[str] = None
    property: Optional[str] = None
    type: Optional[str] = None
    year: Optional[int] = None
    month: Optional[str] = None
    domain: Optional[str] = "all"
    # extras para Compras:
    provider: Optional[str] = None           # mapea a 'proveedor'
    product_keyword: Optional[str] = None    # búsqueda textual en columnas producto/referencia/etc.

class Aggregate(BaseModel):
    metric: Optional[str] = None             # rooms_total | importe_total | unidades_total
    group_by: Optional[List[str]] = None     # p.ej. ["country"], ["property"], ["property","proveedor"]
    weighting: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Filters] = None
    top_k: Optional[int] = 8
    rerank: Optional[bool] = True
    aggregate: Optional[Aggregate] = None
    return_sources: Optional[bool] = True

# ============ NL → JSON (Heurística) ============

def interpret_nl_heuristic(q: str) -> dict:
    if not q:
        return {"query": ""}

    text = q.lower()

    # Habitaciones por país
    if ("habitacion" in text or "habitaciones" in text) and ("país" in text or "pais" in text or "country" in text):
        return {"query": q, "aggregate": {"metric": "rooms_total", "group_by": ["country"]}}

    # Ocupación YTD 2024 (opcional ciudad)
    if ("ocup" in text or "ytd" in text) and ("2024" in text):
        city = None
        m = re.search(r"\ben\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+){0,2})\b", q)
        if m: city = m.group(1).strip()
        payload = {"query": q, "aggregate": {"group_by": ["property"]}, "filters": {"year": 2024}}
        if city: payload["filters"]["city"] = city
        return payload

    # Compras 2023 importe_total por hotel y proveedor
    if ("compra" in text or "compras" in text) and "2023" in text and ("importe" in text or "gasto" in text or "coste" in text):
        return {"query": q, "filters": {"year": 2023}, "aggregate": {"metric": "importe_total", "group_by": ["property","proveedor"]}}

    # Compras: unidades (p.ej., "cuántas toallas compramos en 2025")
    if ("compra" in text or "compramos" in text or "compraste" in text or "comprar" in text) and ("cuant" in text or "cuánt" in text or "unidades" in text):
        # intenta detectar año
        year = None
        ym = re.search(r"(20\d{2})", text)
        if ym:
            try: year = int(ym.group(1))
            except: pass
        # intenta detectar palabra de producto (una palabra después de "cuantas/cuántas")
        prod = None
        pm = re.search(r"(?:cuant[ao]s?|cuánt[ao]s?)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)", text)
        if pm:
            prod = pm.group(1)
        payload = {"query": q, "aggregate": {"metric": "unidades_total"}}
        f = {}
        if year: f["year"] = year
        if prod: f["product_keyword"] = prod
        if f: payload["filters"] = f
        return payload

    return {"query": q}

# ============ NL → JSON (LLM) ============

def call_llm_nl_to_json(q: str) -> Optional[dict]:
    """
    Pide al LLM que devuelva un JSON con:
    {
      "aggregate": {"metric": "...", "group_by": [...]},
      "filters": {"year": 2025, "city": "...", "provider":"...", "product_keyword":"..."}
    }
    """
    if not ENABLE_LLM or not OPENAI_API_KEY or not q.strip():
        return None

    prompt = f"""
Eres un parser que convierte una pregunta de usuario sobre datos hoteleros en un JSON para una API.
La API entiende:
- metric: "rooms_total" | "importe_total" | "unidades_total"
- group_by: lista de campos (p.ej., ["country"], ["property"], ["property","proveedor"]); opcional
- filters: year (int), city (str), country (str), property (str), provider (str), product_keyword (str); opcionales.

Datos disponibles:
- Portfolio: columnas [Marca, Propiedad, Ciudad, País, Tipo, Habitaciones]
- Compras: columnas [referencia_producto, unidades, precio_unitario, precio_total, hotel, fecha (ISO), proveedor]
- Ocupación 2024: columnas [Marca, Propiedad, Ciudad, País, Tipo, YTD, Ene...Dic]

Reglas:
- Si la pregunta pide "cuántas unidades" o "cuántas X compramos", usa metric="unidades_total" y dataset Compras.
- Para Compras, usa filters.year si se menciona año; si se menciona un producto (p.ej., "toallas"), mapea a filters.product_keyword con esa palabra.
- Si piden gasto/importe/coste en Compras, usa metric="importe_total".
- Para ocupación YTD 2024 por propiedad, usa group_by=["property"] y filters.year=2024; si se da ciudad, filters.city.
- Para habitaciones por país, usa metric="rooms_total" y group_by=["country"].

Devuelve SOLO un JSON válido sin texto adicional.
Pregunta: "{q}"
"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Devuelve únicamente JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        # intenta extraer JSON
        json_str = content
        # elimina fences si los hubiese
        if content.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*", "", content)
            json_str = re.sub(r"\s*```$", "", json_str)
        data = json.loads(json_str)
        # sanity-check
        out = {"query": q}
        if isinstance(data, dict):
            if "aggregate" in data and isinstance(data["aggregate"], dict):
                out["aggregate"] = data["aggregate"]
            if "filters" in data and isinstance(data["filters"], dict):
                out["filters"] = data["filters"]
        return out
    except Exception as e:
        print("LLM parse error:", e)
        return None

def interpret_nl_to_queryrequest(q: str) -> dict:
    # 1) LLM si está habilitado
    llm = call_llm_nl_to_json(q)
    if llm: return llm
    # 2) Heurística fallback
    return interpret_nl_heuristic(q)

# ============ HELPERS DE CÁLCULO ============

def compras_apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()
    # renombrar hotel->property para homogeneidad
    if "hotel" in out.columns:
        out = out.rename(columns={"hotel": "property"})
    # año
    if f.year is not None:
        if "fecha" in out.columns:
            out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
            out = out[out["fecha"].dt.year == f.year]
    # proveedor
    if f.provider:
        if "proveedor" in out.columns:
            out = out[out["proveedor"].astype(str).str.contains(f.provider, case=False, na=False)]
    # property
    if f.property:
        if "property" in out.columns:
            out = out[out["property"].astype(str).str.contains(f.property, case=False, na=False)]
    # product_keyword: buscar en posibles columnas de texto
    if f.product_keyword:
        cols = [c for c in ["producto","referencia_producto","descripcion","categoria","concepto"] if c in out.columns]
        if cols:
            mask = False
            for c in cols:
                mask = mask | out[c].astype(str).str.contains(f.product_keyword, case=False, na=False)
            out = out[mask]
    return out

def to_records_sorted(df: pd.DataFrame, by: List[str], ascending=False) -> List[dict]:
    if not by:
        return df.to_dict(orient="records")
    return df.sort_values(by=by, ascending=ascending, na_position="last").to_dict(orient="records")

# ============ ENDPOINTS ============

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

    # Auto-interpretación NL si faltan aggregate/filters
    if (req.aggregate is None) and (req.filters is None) and (req.query or "").strip():
        interpreted = interpret_nl_to_queryrequest(req.query)
        if "aggregate" in interpreted and interpreted["aggregate"]:
            req.aggregate = Aggregate(**interpreted["aggregate"])
        if "filters" in interpreted and interpreted["filters"]:
            req.filters = Filters(**interpreted["filters"])

    f = req.filters or Filters()
    agg = req.aggregate or Aggregate()

    # ---------- Caso Portfolio: rooms_total por country ----------
    if agg.metric == "rooms_total" and (agg.group_by or []) == ["country"]:
        df = DF_PORT.copy()
        df = df.rename(columns={"País": "country", "Habitaciones": "rooms_total"})
        df["rooms_total"] = pd.to_numeric(df["rooms_total"], errors="coerce")
        out = (
            df.groupby("country", dropna=False)["rooms_total"]
            .sum(min_count=1)
            .reset_index()
        )
        return {
            "answer": "Total de habitaciones por país.",
            "aggregates": to_records_sorted(out, by=["rooms_total"], ascending=False),
            "sources": [s for s in source_list() if s["title"] == "Portfolio"],
        }

    # ---------- Caso Ocupación: YTD 2024 por property (opcional city) ----------
    if f.year == 2024 and (agg.group_by or []) == ["property"]:
        df = DF_OCUP.copy()
        df = df.rename(columns={"Propiedad": "property"})
        if f.city:
            df = df[df["Ciudad"].astype(str).str.lower() == f.city.lower()]
        if "YTD" in df.columns:
            df["YTD"] = pd.to_numeric(df["YTD"], errors="coerce")
        out = df[["property","YTD"]].copy()
        return {
            "answer": f"Ranking YTD 2024 por propiedad" + (f" en {f.city}" if f.city else "") + ".",
            "aggregates": to_records_sorted(out, by=["YTD"], ascending=False),
            "sources": [s for s in source_list() if s["title"] == "Ocupación 2024"],
        }

    # ---------- Caso Compras: importe_total o unidades_total ----------
    if agg.metric in {"importe_total", "unidades_total"}:
        df = DF_COMP.copy()
        # normalizaciones
        if "precio_total" in df.columns:
            df["importe_total"] = pd.to_numeric(df["precio_total"], errors="coerce")
        if "unidades" in df.columns:
            df["unidades"] = pd.to_numeric(df["unidades"], errors="coerce")

        df = compras_apply_filters(df, f)

        # Agrupación dinámica
        group = list(agg.group_by or [])
        # map 'property' and 'proveedor' to existing cols
        rename_map = {}
        if "hotel" in df.columns and "property" not in df.columns:
            rename_map["hotel"] = "property"
        if rename_map:
            df = df.rename(columns=rename_map)

        # Si no hay group_by => total
        if not group:
            if agg.metric == "importe_total":
                total = float(pd.to_numeric(df.get("importe_total", pd.Series(dtype=float)), errors="coerce").sum())
                return {
                    "answer": "Importe total de compras" + (f" en {f.year}" if f.year else "") + ".",
                    "aggregates": [{"importe_total": total}],
                    "sources": [s for s in source_list() if s["title"] == "Compras"],
                }
            else:
                total = float(pd.to_numeric(df.get("unidades", pd.Series(dtype=float)), errors="coerce").sum())
                return {
                    "answer": "Unidades totales compradas" + (f" en {f.year}" if f.year else "") + (f" del producto '{f.product_keyword}'" if f.product_keyword else "") + ".",
                    "aggregates": [{"unidades_total": total}],
                    "sources": [s for s in source_list() if s["title"] == "Compras"],
                }

        # Con group_by
        cols_check = []
        for g in group:
            if g == "proveedor":
                cols_check.append("proveedor")
            elif g == "property":
                cols_check.append("property")
            else:
                cols_check.append(g)
        # filtra a columnas existentes
        cols_group = [c for c in cols_check if c in df.columns]
        if not cols_group:
            # si el group_by no existe en df, respondemos totales simples
            if agg.metric == "importe_total":
                total = float(pd.to_numeric(df.get("importe_total", pd.Series(dtype=float)), errors="coerce").sum())
                return {
                    "answer": "Importe total de compras (sin columnas de agrupación válidas).",
                    "aggregates": [{"importe_total": total}],
                    "sources": [s for s in source_list() if s["title"] == "Compras"],
                }
            else:
                total = float(pd.to_numeric(df.get("unidades", pd.Series(dtype=float)), errors="coerce").sum())
                return {
                    "answer": "Unidades totales compradas (sin columnas de agrupación válidas).",
                    "aggregates": [{"unidades_total": total}],
                    "sources": [s for s in source_list() if s["title"] == "Compras"],
                }

        if agg.metric == "importe_total":
            g = df.groupby(cols_group, dropna=False)["importe_total"].sum(min_count=1).reset_index()
            return {
                "answer": "Importe total de compras por " + ", ".join(cols_group) + (f" en {f.year}" if f.year else "") + ".",
                "aggregates": to_records_sorted(g, by=["importe_total"], ascending=False),
                "sources": [s for s in source_list() if s["title"] == "Compras"],
            }
        else:
            g = df.groupby(cols_group, dropna=False)["unidades"].sum(min_count=1).reset_index()
            return {
                "answer": "Unidades totales compradas por " + ", ".join(cols_group) + (f" en {f.year}" if f.year else "") + (f" del producto '{f.product_keyword}'" if f.product_keyword else "") + ".",
                "aggregates": to_records_sorted(g, by=["unidades"], ascending=False),
                "sources": [s for s in source_list() if s["title"] == "Compras"],
            }

    # ---------- Fallback ----------
    return {
        "answer": "Consulta no reconocida. Prueba ejemplos: 'Total de habitaciones por país', 'Ocupación YTD 2024 por propiedad en <ciudad>', 'Coste total de compras 2023 por hotel y proveedor', '¿Cuántas toallas compramos en 2025?'.",
        "aggregates": [],
        "sources": source_list(),
    }

# ============ STARTUP ============

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
