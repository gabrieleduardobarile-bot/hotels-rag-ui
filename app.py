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

# ============ NL → JSON (LLM con System Message + few-shots) ============

def normalize_llm_json(q: str, data: dict) -> dict:
    """
    Sanitiza la salida del LLM para ajustarse al contrato de la API:
      - Solo claves permitidas
      - Métricas válidas: rooms_total | importe_total | unidades_total | ocupacion_media
      - group_by si es lista de strings
      - filters con campos conocidos y tipos razonables
    """
    allowed_metrics = {"rooms_total", "importe_total", "unidades_total", "ocupacion_media"}
    allowed_filters = {"year","city","country","property","provider","product_keyword","month","brand","type","domain"}
    out = {"query": q}

    # aggregate
    agg = data.get("aggregate") if isinstance(data, dict) else None
    if isinstance(agg, dict):
        metric = agg.get("metric")
        group_by = agg.get("group_by")
        norm_agg = {}
        if isinstance(metric, str) and metric in allowed_metrics:
            norm_agg["metric"] = metric
        if isinstance(group_by, list) and all(isinstance(x, str) for x in group_by):
            uniq = []
            for g in group_by:
                g = g.strip()
                if g and g not in uniq:
                    uniq.append(g)
            if uniq:
                norm_agg["group_by"] = uniq
        if norm_agg:
            out["aggregate"] = norm_agg

    # filters
    f = data.get("filters") if isinstance(data, dict) else None
    if isinstance(f, dict):
        norm_f = {}
        for k, v in f.items():
            if k in allowed_filters:
                if k == "year":
                    try:
                        norm_f["year"] = int(v)
                    except:
                        pass
                else:
                    if isinstance(v, (str, int, float)):
                        norm_f[k] = str(v)
        if norm_f:
            out["filters"] = norm_f

    return out

def call_llm_nl_to_json(q: str) -> Optional[dict]:
    """
    Pide al LLM un JSON con:
      {
        "aggregate": {"metric": "...", "group_by": [...]},   # metric ∈ {rooms_total, importe_total, unidades_total, ocupacion_media}
        "filters": {"year": 2025, "city": "...", "country":"...", "property":"...", "provider":"...", "product_keyword":"..."}
      }
    """
    if not ENABLE_LLM or not OPENAI_API_KEY or not q.strip():
        return None

    system_msg = """
Eres un parser experto en BI hotelero. Tu única tarea es transformar una pregunta en ESPAÑOL a un JSON válido
que sirva para una API analítica tabular. NO expliques, NO añadas texto, NO pongas bloques de código; devuelve
EXCLUSIVAMENTE un JSON. Reglas y contexto:

1) Esquema de datos disponible:
   - Portfolio: [Marca, Propiedad, Ciudad, País, Tipo, Habitaciones]
   - Compras: [referencia_producto, unidades, precio_unitario, precio_total, hotel, fecha (ISO), proveedor]
   - Ocupación 2024: [Marca, Propiedad, Ciudad, País, Tipo, YTD, Ene...Dic]
   No inventes columnas. No inventes datasets.

2) Salida esperada (JSON):
   {
     "aggregate": {
       "metric": "rooms_total" | "importe_total" | "unidades_total" | "ocupacion_media",
       "group_by": [string, ...]  // opcional; ejemplos: ["country"], ["property"], ["property","proveedor"]
     },
     "filters": {
       "year": int,                // opcional
       "city": string,             // opcional
       "country": string,          // opcional
       "property": string,         // opcional (hotel)
       "provider": string,         // opcional (mapear “proveedor”)
       "product_keyword": string   // opcional (palabra clave del producto a buscar en texto)
     }
   }

3) Reglas de negocio:
   - “Habitaciones por país” → metric="rooms_total", group_by=["country"].
   - “Ocupación promedio/media” → metric="ocupacion_media". Si mencionan ciudad, incluye filters.city.
     Si no especifican año, asume que el dataset de ocupación es YTD 2024 (no es necesario incluir year).
   - “Ocupación YTD 2024 por propiedad en <ciudad>” → group_by=["property"], filters.year=2024, filters.city=<ciudad>.
   - “Gasto/importe/coste de compras” → metric="importe_total". Añade filters.year si lo mencionan explícitamente.
     Si piden por proveedor u hotel, añade group_by=["proveedor"] o ["property"] (o ambos).
   - “¿Cuántas <producto> compramos ...?” o “¿cuántas unidades ...?” → metric="unidades_total".
     Si se menciona año, filters.year; producto, product_keyword con esa palabra (p.ej., “toallas”).
   - Si no hay suficiente información, devuelve el mejor JSON posible sin inventar.

4) Formato:
   - Devuelve solo JSON plano sin comentarios, sin fences, sin texto adicional.
   - Usa claves y valores exactos según las reglas; no devuelvas campos desconocidos.
    """.strip()

    fewshots = [
        {
            "user": "habitaciones por país",
            "assistant": {"aggregate": {"metric": "rooms_total", "group_by": ["country"]}}
        },
        {
            "user": "ocupación promedio en Barcelona",
            "assistant": {"aggregate": {"metric": "ocupacion_media"}, "filters": {"city": "Barcelona"}}
        },
        {
            "user": "ocupación ytd 2024 por propiedad en Madrid",
            "assistant": {"aggregate": {"group_by": ["property"]}, "filters": {"year": 2024, "city": "Madrid"}}
        },
        {
            "user": "importe total de compras 2023 por proveedor",
            "assistant": {"aggregate": {"metric": "importe_total", "group_by": ["proveedor"]}, "filters": {"year": 2023}}
        },
        {
            "user": "cuántas toallas compramos en 2025",
            "assistant": {"aggregate": {"metric": "unidades_total"}, "filters": {"year": 2025, "product_keyword": "toallas"}}
        },
    ]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        messages = [{"role": "system", "content": system_msg}]
        for fs in fewshots:
            messages.append({"role": "user", "content": fs["user"]})
            messages.append({"role": "assistant", "content": json.dumps(fs["assistant"], ensure_ascii=False)})

        messages.append({"role": "user", "content": q})

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()

        json_str = content
        if content.startswith("```"):
            json_str = re.sub(r"^```(?:json)?\s*", "", content)
            json_str = re.sub(r"\s*```$", "", json_str)

        data = json.loads(json_str)
        return normalize_llm_json(q, data)
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
    # producto: buscar en TODAS las columnas de texto (por si no existe una columna específica)
    if f.product_keyword:
        text_cols = list(out.select_dtypes(include=["object"]).columns)
        if text_cols:
            mask = False
            for c in text_cols:
                mask = (mask | out[c].astype(str).str.contains(f.product_keyword, case=False, na=False))
            matched = out[mask]
            if len(matched) > 0:
                out = matched
            # si no hay coincidencias, mantenemos out sin filtrar y avisamos en la respuesta
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

        before_rows = len(df)
        df = compras_apply_filters(df, f)
        after_rows = len(df)

        note = None
        if f.product_keyword and after_rows == before_rows:
            note = f"No se encontraron coincidencias para '{f.product_keyword}' en columnas de texto; devolviendo totales sin filtro de producto."

        group = list(agg.group_by or [])
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
