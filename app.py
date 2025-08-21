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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============ APP INIT ============

app = FastAPI(title="Hotels RAG Demo (NL→JSON + LLM)")

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
    # extras Compras
    provider: Optional[str] = None            # mapea a 'proveedor'
    product_keyword: Optional[str] = None     # búsqueda textual amplia

class Aggregate(BaseModel):
    # soportadas: rooms_total | importe_total | unidades_total | ocupacion_media
    metric: Optional[str] = None
    group_by: Optional[List[str]] = None
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

    # Ocupación promedio (media) — por ciudad opcional
    if ("ocup" in text or "ocupación" in text) and ("promedio" in text or "media" in text):
        city = None
        m = re.search(r"\ben\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+){0,2})\b", q)
        if m: city = m.group(1).strip()
        payload = {"query": q, "aggregate": {"metric": "ocupacion_media"}}
        if city:
            payload["filters"] = {"city": city}
        return payload

    # Ocupación YTD 2024 (ranking por propiedad en ciudad opcional)
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

    # Compras: unidades (cuántas X compramos en <año>)
    if ("compra" in text or "compramos" in text or "compraste" in text or "comprar" in text) and (("cuant" in text) or ("cuánt" in text) or ("unidades" in text)):
        year = None
        ym = re.search(r"(20\d{2})", text)
        if ym:
            try: year = int(ym.group(1))
            except: pass
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
    Pide al LLM que devuelva JSON con:
      aggregate.metric ∈ {"rooms_total","importe_total","unidades_total","ocupacion_media"}
      aggregate.group_by opcional
      filters: year, city, country, property, provider, product_keyword
    Reglas: usa ocupacion_media si piden 'promedio' o 'media' de ocupación, asume 2024 si no indican año.
    """
    if not ENABLE_LLM or not OPENAI_API_KEY or not q.strip():
        return None

    prompt = f"""
Eres un parser que convierte la pregunta del usuario en JSON para una API.
- metric: "rooms_total" | "importe_total" | "unidades_total" | "ocupacion_media"
- group_by: lista de campos (p.ej., ["country"], ["property"], ["property","proveedor"]); opcional.
- filters: year (int), city (str), country (str), property (str), provider (str), product_keyword (str); opcionales.
Datos:
- Portfolio: [Marca, Propiedad, Ciudad, País, Tipo, Habitaciones]
- Compras: [referencia_producto, unidades, precio_unitario, precio_total, hotel, fecha, proveedor]
- Ocupación 2024: [Marca, Propiedad, Ciudad, País, Tipo, YTD, Ene...Dic]
Reglas:
- "cuántas X compramos" ⇒ metric="unidades_total", filters.year si se indica, product_keyword con esa X.
- "gasto/importe de compras" ⇒ metric="importe_total".
- "ocupación promedio/media" ⇒ metric="ocupacion_media"; si hay ciudad, filters.city. Si no hay año, asume 2024.
- "habitaciones por país" ⇒ metric="rooms_total", group_by=["country"].
Devuelve SOLO JSON.
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
        json_str = content
        if content.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*", "", content)
            json_str = re.sub(r"\s*```$", "", json_str)
        data = json.loads(json_str)
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
    llm = call_llm_nl_to_json(q)
    if llm:
        return llm
    return interpret_nl_heuristic(q)

# ============ HELPERS DE CÁLCULO ============

def compras_apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df.copy()
    if "hotel" in out.columns and "property" not in out.columns:
        out = out.rename(columns={"hotel": "property"})
    # año
    if f.year is not None and "fecha" in out.columns:
        out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
        out = out[out["fecha"].dt.year == f.year]
    # proveedor
    if f.provider and "proveedor" in out.columns:
        out = out[out["proveedor"].astype(str).str.contains(f.provider, case=False, na=False)]
    # property
    if f.property and "property" in out.columns:
        out = out[out["property"].astype(str).str.contains(f.property, case=False, na=False)]
    # producto: buscar en TODAS las columnas de texto
    if f.product_keyword:
        text_cols = list(out.select_dtypes(include=["object"]).columns)
        if text_cols:
            mask = False
            for c in text_cols:
                mask = (mask | out[c].astype(str).str.contains(f.product_keyword, case=False, na=False))
            matched = out[mask]
            if len(matched) > 0:
                out = matched
            # si no hay coincidencias, dejamos out tal cual y lo indicaremos en la respuesta
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

    # Autointerpretación desde NL si faltan aggregate/filters
    if (req.aggregate is None) and (req.filters is None) and (req.query or "").strip():
        interpreted = interpret_nl_to_queryrequest(req.query)
        if "aggregate" in interpreted and interpreted["aggregate"]:
            req.aggregate = Aggregate(**interpreted["aggregate"])
        if "filters" in interpreted and interpreted["filters"]:
            req.filters = Filters(**interpreted["filters"])

    f = req.filters or Filters()
    agg = req.aggregate or Aggregate()

    # ---------- Habitaciones por país ----------
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

    # ---------- Ocupación promedio (media) ----------
    if agg.metric == "ocupacion_media":
        df = DF_OCUP.copy()
        # por claridad, no exigimos year; asumimos YTD 2024 del dataset
        if f.city:
            df = df[df["Ciudad"].astype(str).str.lower() == f.city.lower()]
        if "YTD" in df.columns:
            df["YTD"] = pd.to_numeric(df["YTD"], errors="coerce")
            mean_val = float(df["YTD"].mean()) if len(df) else None
        else:
            mean_val = None
        aggregates = []
        if mean_val is not None:
            record = {"ocupacion_media": round(mean_val, 2)}
            if f.city: record["city"] = f.city
            aggregates.append(record)
        return {
            "answer": ("Ocupación promedio (YTD) " + (f"en {f.city}" if f.city else "global") + "."),
            "aggregates": aggregates,
            "sources": [s for s in source_list() if s["title"] == "Ocupación 2024"],
        }

    # ---------- Ocupación: ranking YTD 2024 por propiedad (opcional ciudad) ----------
    if (agg.group_by or []) == ["property"]:
        df = DF_OCUP.copy()
        df = df.rename(columns={"Propiedad": "property"})
        if f.city:
            df = df[df["Ciudad"].astype(str).str.lower() == f.city.lower()]
        if "YTD" in df.columns:
            df["YTD"] = pd.to_numeric(df["YTD"], errors="coerce")
        out = df[["property", "YTD"]].copy()
        return {
            "answer": f"Ranking YTD por propiedad" + (f" en {f.city}" if f.city else "") + ".",
            "aggregates": to_records_sorted(out, by=["YTD"], ascending=False),
            "sources": [s for s in source_list() if s["title"] == "Ocupación 2024"],
        }

    # ---------- Compras: importe_total o unidades_total ----------
    if agg.metric in {"importe_total", "unidades_total"}:
        df = DF_COMP.copy()
        if "precio_total" in df.columns:
            df["importe_total"] = pd.to_numeric(df["precio_total"], errors="coerce")
        if "unidades" in df.columns:
            df["unidades"] = pd.to_numeric(df["unidades"], errors="coerce")

        # copia para chequear si hubo coincidencias por product_keyword
        before_rows = len(df)
        df = compras_apply_filters(df, f)
        after_rows = len(df)

        note = None
        if f.product_keyword and after_rows == before_rows:
            note = f"No se encontraron coincidencias para '{f.product_keyword}' en columnas de texto; devolviendo totales sin filtro de producto."

        group = list(agg.group_by or [])
        # renombrar a 'property' si procede
        if "hotel" in df.columns and "property" not in df.columns:
            df = df.rename(columns={"hotel": "property"})

        # Sin group_by => totales
        if not group:
            if agg.metric == "importe_total":
                total = float(pd.to_numeric(df.get("importe_total", pd.Series(dtype=float)), errors="coerce").sum())
                ans = "Importe total de compras" + (f" en {f.year}" if f.year else "") + "."
                if note: ans += " " + note
                return {
                    "answer": ans,
                    "aggregates": [{"importe_total": total}],
                    "sources": [s for s in source_list() if s["title"] == "Compras"],
                }
            else:
                total = float(pd.to_numeric(df.get("unidades", pd.Series(dtype=float)), errors="coerce").sum())
                ans = "Unidades totales compradas" + (f" en {f.year}" if f.year else "") + (f" del producto '{f.product_keyword}'" if f.product_keyword else "") + "."
                if note: ans += " " + note
                return {
                    "answer": ans,
                    "aggregates": [{"unidades_total": total}],
                    "sources": [s for s in source_list() if s["title"] == "Compras"],
                }

        # Con group_by
        cols_check = []
        for g in group:
            if g == "proveedor": cols_check.append("proveedor")
            elif g == "property": cols_check.append("property")
            else: cols_check.append(g)
        cols_group = [c for c in cols_check if c in df.columns]

        if not cols_group:
            # si no existen columnas de agrupación pedidas
            if agg.metric == "importe_total":
                total = float(pd.to_numeric(df.get("importe_total", pd.Series(dtype=float)), errors="coerce").sum())
                ans = "Importe total de compras (sin columnas de agrupación válidas)."
                if note: ans += " " + note
                return {"answer": ans, "aggregates": [{"importe_total": total}], "sources": [s for s in source_list() if s["title"] == "Compras"]}
            else:
                total = float(pd.to_numeric(df.get("unidades", pd.Series(dtype=float)), errors="coerce").sum())
                ans = "Unidades totales compradas (sin columnas de agrupación válidas)."
                if note: ans += " " + note
                return {"answer": ans, "aggregates": [{"unidades_total": total}], "sources": [s for s in source_list() if s["title"] == "Compras"]}

        if agg.metric == "importe_total":
            g = df.groupby(cols_group, dropna=False)["importe_total"].sum(min_count=1).reset_index()
            ans = "Importe total de compras por " + ", ".join(cols_group) + (f" en {f.year}" if f.year else "") + "."
            if note: ans += " " + note
            return {
                "answer": ans,
                "aggregates": to_records_sorted(g, by=["importe_total"], ascending=False),
                "sources": [s for s in source_list() if s["title"] == "Compras"],
            }
        else:
            g = df.groupby(cols_group, dropna=False)["unidades"].sum(min_count=1).reset_index()
            ans = "Unidades totales compradas por " + ", ".join(cols_group) + (f" en {f.year}" if f.year else "") + (f" del producto '{f.product_keyword}'" if f.product_keyword else "") + "."
            if note: ans += " " + note
            return {
                "answer": ans,
                "aggregates": to_records_sorted(g, by=["unidades"], ascending=False),
                "sources": [s for s in source_list() if s["title"] == "Compras"],
            }

    # ---------- Fallback ----------
    return {
        "answer": "Consulta no reconocida. Ejemplos: 'Ocupación promedio en Barcelona', '¿Cuántas toallas compramos en 2024?', 'Total de habitaciones por país'.",
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
