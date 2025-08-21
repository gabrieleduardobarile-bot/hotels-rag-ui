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

app = FastAPI(title="Hotels RAG Demo (LLM function-calling + ranking)")

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

def current_schema_context() -> str:
    try:
        cols_port = list(DF_PORT.columns) if DF_PORT is not None else []
        cols_comp = list(DF_COMP.columns) if DF_COMP is not None else []
        cols_ocup = list(DF_OCUP.columns) if DF_OCUP is not None else []
    except Exception:
        cols_port, cols_comp, cols_ocup = [], [], []
    return (
        "Columnas reales disponibles:\n"
        f"- Portfolio: {cols_port or ['(desconocidas)']}\n"
        f"- Compras: {cols_comp or ['(desconocidas)']}\n"
        f"- Ocupación 2024: {cols_ocup or ['(desconocidas)']}\n"
        "Usa solo estas columnas lógicas en group_by/filters: country, city, property, proveedor(provider), product_keyword.\n"
        "No inventes columnas."
    )

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
    provider: Optional[str] = None
    product_keyword: Optional[str] = None

class Aggregate(BaseModel):
    # métricas soportadas
    metric: Optional[str] = None  # rooms_total | importe_total | unidades_total | ocupacion_media
    group_by: Optional[List[str]] = None
    weighting: Optional[str] = None
    # NUEVO: orden y límite
    order_by: Optional[str] = None   # "YTD" | "importe_total" | "unidades" | "rooms_total"
    order: Optional[str] = None      # "asc" | "desc"
    limit: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Filters] = None
    top_k: Optional[int] = 8
    rerank: Optional[bool] = True
    aggregate: Optional[Aggregate] = None
    return_sources: Optional[bool] = True

# ============ NL → JSON (Heurística) ============

def _extract_int(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", text)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def interpret_nl_heuristic(q: str) -> dict:
    if not q:
        return {"query": ""}
    text = q.lower()

    # Habitaciones por país
    if ("habitacion" in text or "habitaciones" in text) and ("país" in text or "pais" in text or "country" in text):
        return {"query": q, "aggregate": {"metric": "rooms_total", "group_by": ["country"], "order_by":"rooms_total", "order":"desc"}}

    # Ranking ocupación (top/bottom N hoteles) - asume 2024 si no dicen año
    if ("ocup" in text or "ocupación" in text) and ("hotel" in text or "hoteles" in text):
        limit = _extract_int(text)
        order = "asc" if "menos" in text or "baja" in text else ("desc" if "más" in text or "alta" in text else None)
        payload = {"query": q, "aggregate": {"group_by": ["property"], "order_by": "YTD"} , "filters": {"year": 2024}}
        if order:
            payload["aggregate"]["order"] = order
        if limit:
            payload["aggregate"]["limit"] = limit
        return payload

    # Ocupación promedio (media) — ciudad opcional
    if ("ocup" in text or "ocupación" in text) and ("promedio" in text or "media" in text):
        city = None
        m = re.search(r"\ben\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+){0,2})\b", q)
        if m: city = m.group(1).strip()
        payload = {"query": q, "aggregate": {"metric": "ocupacion_media"}}
        if city: payload["filters"] = {"city": city}
        return payload

    # Ocupación YTD 2024 por propiedad en ciudad
    if ("ocup" in text or "ytd" in text) and ("2024" in text):
        city = None
        m = re.search(r"\ben\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+){0,2})\b", q)
        if m: city = m.group(1).strip()
        payload = {"query": q, "aggregate": {"group_by": ["property"], "order_by":"YTD", "order":"desc"}, "filters": {"year": 2024}}
        if city: payload["filters"]["city"] = city
        return payload

    # Compras 2023 importe_total por hotel y proveedor
    if ("compra" in text or "compras" in text) and "2023" in text and ("importe" in text or "gasto" in text or "coste" in text):
        return {"query": q, "filters": {"year": 2023}, "aggregate": {"metric": "importe_total", "group_by": ["property","proveedor"], "order_by":"importe_total", "order":"desc"}}

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
        payload = {"query": q, "aggregate": {"metric": "unidades_total", "order_by":"unidades", "order":"desc"}}
        f = {}
        if year: f["year"] = year
        if prod: f["product_keyword"] = prod
        if f: payload["filters"] = f
        return payload

    return {"query": q}

# ============ NL → JSON (LLM function-calling + System Message) ============

def normalize_llm_json(q: str, data: dict) -> dict:
    allowed_metrics = {"rooms_total", "importe_total", "unidades_total", "ocupacion_media"}
    allowed_filters = {"year","city","country","property","provider","product_keyword","month","brand","type","domain"}
    allowed_order = {"asc","desc"}
    out = {"query": q}

    # aggregate
    agg = data.get("aggregate") if isinstance(data, dict) else None
    if isinstance(agg, dict):
        norm_agg = {}
        metric = agg.get("metric")
        if isinstance(metric, str) and metric in allowed_metrics:
            norm_agg["metric"] = metric
        group_by = agg.get("group_by")
        if isinstance(group_by, list) and all(isinstance(x, str) for x in group_by):
            uniq = []
            for g in group_by:
                g = g.strip()
                if g and g not in uniq: uniq.append(g)
            if uniq: norm_agg["group_by"] = uniq
        order_by = agg.get("order_by")
        if isinstance(order_by, str) and order_by.strip():
            norm_agg["order_by"] = order_by.strip()
        order = agg.get("order")
        if isinstance(order, str) and order.lower() in allowed_order:
            norm_agg["order"] = order.lower()
        limit = agg.get("limit")
        if isinstance(limit, int) and limit > 0:
            norm_agg["limit"] = limit
        if norm_agg: out["aggregate"] = norm_agg

    # filters
    f = data.get("filters") if isinstance(data, dict) else None
    if isinstance(f, dict):
        norm_f = {}
        for k, v in f.items():
            if k in allowed_filters:
                if k == "year":
                    try: norm_f["year"] = int(v)
                    except: pass
                else:
                    if isinstance(v, (str, int, float)): norm_f[k] = str(v)
        if norm_f: out["filters"] = norm_f

    return out

def call_llm_nl_to_json(q: str) -> Optional[dict]:
    if not ENABLE_LLM or not OPENAI_API_KEY or not q.strip():
        return None

    ensure_loaded()
    schema_ctx = current_schema_context()

    system_msg = f"""
Eres un parser experto en BI hotelero. Transforma una pregunta en ESPAÑOL en JSON para una API analítica.
NO expliques, NO añadas texto; usa function-calling con la función 'to_query'.
Contexto:
- Datasets: Portfolio, Compras, Ocupación 2024 (YTD). No inventes datasets ni columnas.
- Mapea a claves semánticas: country, city, property (hotel), proveedor (provider), product_keyword.
- Reglas:
  * "Habitaciones por país" → metric=rooms_total, group_by=["country"], order_by="rooms_total", order="desc".
  * "Ocupación promedio/media (en <ciudad>)" → metric=ocupacion_media (+ filters.city).
  * "Top/Bottom N hoteles por ocupación (2024 por defecto)" → group_by=["property"], filters.year=2024,
    order_by="YTD", order="desc" para "más" / "asc" para "menos", limit=N si se indica.
  * "Ocupación YTD 2024 por propiedad en <ciudad>" → group_by=["property"], filters.year=2024, filters.city=<ciudad>, order_by="YTD", order="desc".
  * "Gasto/importe de compras" → metric=importe_total (+ year si se indica; usa group_by si piden por proveedor/property);
    siempre que agregues, añade order_by="importe_total" con order="desc".
  * "¿Cuántas <producto> compramos (en <año>)?" → metric=unidades_total (+ filters.year; product_keyword=<producto>), order_by="unidades", order="desc".
{schema_ctx}
""".strip()

    fewshots = [
        ("habitaciones por país",
         {"aggregate": {"metric": "rooms_total", "group_by": ["country"], "order_by":"rooms_total", "order":"desc"}}),
        ("ocupación promedio en Barcelona",
         {"aggregate": {"metric": "ocupacion_media"}, "filters": {"city": "Barcelona"}}),
        ("ocupación ytd 2024 por propiedad en Madrid",
         {"aggregate": {"group_by": ["property"], "order_by":"YTD", "order":"desc"}, "filters": {"year": 2024, "city": "Madrid"}}),
        ("importe total de compras 2023 por proveedor",
         {"aggregate": {"metric": "importe_total", "group_by": ["proveedor"], "order_by":"importe_total", "order":"desc"}, "filters": {"year": 2023}}),
        ("cuántas toallas compramos en 2025",
         {"aggregate": {"metric": "unidades_total", "order_by":"unidades", "order":"desc"}, "filters": {"year": 2025, "product_keyword": "toallas"}}),
        ("cuales fueron los 3 hoteles con menos ocupacion del 2024",
         {"aggregate": {"group_by":["property"], "order_by":"YTD", "order":"asc", "limit":3}, "filters":{"year":2024}})
    ]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        tools = [{
            "type": "function",
            "function": {
                "name": "to_query",
                "description": "Convierte la pregunta del usuario en parámetros para la API analítica.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "aggregate": {
                            "type": "object",
                            "properties": {
                                "metric": {"type": "string", "enum": ["rooms_total","importe_total","unidades_total","ocupacion_media"]},
                                "group_by": {"type": "array", "items": {"type": "string"}},
                                "order_by": {"type": "string"},
                                "order": {"type": "string", "enum": ["asc","desc"]},
                                "limit": {"type": "integer", "minimum": 1}
                            },
                            "additionalProperties": False
                        },
                        "filters": {
                            "type": "object",
                            "properties": {
                                "year": {"type": "integer"},
                                "city": {"type": "string"},
                                "country": {"type": "string"},
                                "property": {"type": "string"},
                                "provider": {"type": "string"},
                                "product_keyword": {"type": "string"},
                                "month": {"type": "string"},
                                "brand": {"type": "string"},
                                "type": {"type": "string"},
                                "domain": {"type": "string"}
                            },
                            "additionalProperties": False
                        }
                    },
                    "additionalProperties": False
                }
            }
        }]

        messages = [{"role": "system", "content": system_msg}]
        for u, a in fewshots:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})

        messages.append({"role": "user", "content": q})

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "to_query"}},
            temperature=0.1,
        )

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and len(tool_calls) > 0:
            call = tool_calls[0]
            args_str = call.function.arguments or "{}"
            data = json.loads(args_str)
            return normalize_llm_json(q, data)

        content = (msg.content or "").strip()
        if content:
            c = content
            if c.startswith("```"):
                c = re.sub(r"^```(?:json)?\s*", "", c); c = re.sub(r"\s*```$", "", c)
            data = json.loads(c)
            return normalize_llm_json(q, data)

        return None
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
    if f.year is not None and "fecha" in out.columns:
        out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
        out = out[out["fecha"].dt.year == f.year]
    if f.provider and "proveedor" in out.columns:
        out = out[out["proveedor"].astype(str).str.contains(f.provider, case=False, na=False)]
    if f.property and "property" in out.columns:
        out = out[out["property"].astype(str).str.contains(f.property, case=False, na=False)]
    if f.product_keyword:
        text_cols = list(out.select_dtypes(include=["object"]).columns)
        if text_cols:
            mask = False
            for c in text_cols:
                mask = (mask | out[c].astype(str).str.contains(f.product_keyword, case=False, na=False))
            matched = out[mask]
            if len(matched) > 0:
                out = matched
    return out

def to_records_sorted_limited(df: pd.DataFrame, by: Optional[str], order: Optional[str], limit: Optional[int]):
    if by and by in df.columns:
        df = df.sort_values(by=by, ascending=(order=="asc"), na_position="last")
    if isinstance(limit, int) and limit > 0:
        df = df.head(limit)
    return df.to_dict(orient="records")

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
        df = DF_PORT.copy().rename(columns={"País": "country", "Habitaciones": "rooms_total"})
        df["rooms_total"] = pd.to_numeric(df["rooms_total"], errors="coerce")
        out = df.groupby("country", dropna=False)["rooms_total"].sum(min_count=1).reset_index()
        records = to_records_sorted_limited(out, by=agg.order_by or "rooms_total", order=agg.order or "desc", limit=agg.limit)
        return {"answer": "Total de habitaciones por país.", "aggregates": records, "sources": [s for s in source_list() if s["title"]=="Portfolio"]}

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
        return {"answer": ("Ocupación promedio (YTD) " + (f"en {f.city}" if f.city else "global") + "."), "aggregates": aggregates, "sources": [s for s in source_list() if s["title"]=="Ocupación 2024"]}

    # ---------- Ocupación: ranking por propiedad (YTD 2024 por defecto si se pide ranking) ----------
    if (agg.group_by or []) == ["property"]:
        df = DF_OCUP.copy().rename(columns={"Propiedad":"property"})
        # Si piden ranking y no especifican año, asumimos 2024 (dataset actual)
        if f.city:
            df = df[df["Ciudad"].astype(str).str.lower() == f.city.lower()]
        if "YTD" in df.columns:
            df["YTD"] = pd.to_numeric(df["YTD"], errors="coerce")
        out = df[["property","YTD"]].copy()
        records = to_records_sorted_limited(out, by=agg.order_by or "YTD", order=agg.order or "desc", limit=agg.limit)
        ans = f"Ranking YTD por propiedad" + (f" en {f.city}" if f.city else "") + "."
        return {"answer": ans, "aggregates": records, "sources": [s for s in source_list() if s["title"]=="Ocupación 2024"]}

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
            note = f"No hubo coincidencias para '{f.product_keyword}' en columnas de texto; se devuelven totales sin ese filtro."

        # map hotel->property si procede
        if "hotel" in df.columns and "property" not in df.columns:
            df = df.rename(columns={"hotel":"property"})

        group = list(agg.group_by or [])
        if not group:
            if agg.metric == "importe_total":
                total = float(pd.to_numeric(df.get("importe_total", pd.Series(dtype=float)), errors="coerce").sum())
                ans = "Importe total de compras" + (f" en {f.year}" if f.year else "") + "."
                if note: ans += " " + note
                return {"answer": ans, "aggregates": [{"importe_total": total}], "sources": [s for s in source_list() if s["title"]=="Compras"]}
            else:
                total = float(pd.to_numeric(df.get("unidades", pd.Series(dtype=float)), errors="coerce").sum())
                ans = "Unidades totales compradas" + (f" en {f.year}" if f.year else "") + (f" del producto '{f.product_keyword}'" if f.product_keyword else "") + "."
                if note: ans += " " + note
                return {"answer": ans, "aggregates": [{"unidades_total": total}], "sources": [s for s in source_list() if s["title"]=="Compras"]}

        # con group_by
        cols_check = []
        for g in group:
            if g == "proveedor": cols_check.append("proveedor")
            elif g == "property": cols_check.append("property")
            else: cols_check.append(g)
        cols_group = [c for c in cols_check if c in df.columns]

        if not cols_group:
            if agg.metric == "importe_total":
                total = float(pd.to_numeric(df.get("importe_total", pd.Series(dtype=float)), errors="coerce").sum())
                ans = "Importe total de compras (sin columnas válidas de agrupación)."
                if note: ans += " " + note
                return {"answer": ans, "aggregates": [{"importe_total": total}], "sources": [s for s in source_list() if s["title"]=="Compras"]}
            else:
                total = float(pd.to_numeric(df.get("unidades", pd.Series(dtype=float)), errors="coerce").sum())
                ans = "Unidades totales compradas (sin columnas válidas de agrupación)."
                if note: ans += " " + note
                return {"answer": ans, "aggregates": [{"unidades_total": total}], "sources": [s for s in source_list() if s["title"]=="Compras"]}

        if agg.metric == "importe_total":
            g = df.groupby(cols_group, dropna=False)["importe_total"].sum(min_count=1).reset_index()
            records = to_records_sorted_limited(g, by=agg.order_by or "importe_total", order=agg.order or "desc", limit=agg.limit)
            ans = "Importe total de compras por " + ", ".join(cols_group) + (f" en {f.year}" if f.year else "") + "."
            if note: ans += " " + note
            return {"answer": ans, "aggregates": records, "sources": [s for s in source_list() if s["title"]=="Compras"]}
        else:
            g = df.groupby(cols_group, dropna=False)["unidades"].sum(min_count=1).reset_index()
            records = to_records_sorted_limited(g, by=agg.order_by or "unidades", order=agg.order or "desc", limit=agg.limit)
            ans = "Unidades totales compradas por " + ", ".join(cols_group) + (f" en {f.year}" if f.year else "") + (f" del producto '{f.product_keyword}'" if f.product_keyword else "") + "."
            if note: ans += " " + note
            return {"answer": ans, "aggregates": records, "sources": [s for s in source_list() if s["title"]=="Compras"]}

    # ---------- Fallback ----------
    return {
        "answer": "Consulta no reconocida. Ejemplos: '¿Cuáles fueron los 3 hoteles con menos ocupación de 2024?', 'Ocupación promedio en Barcelona', 'Total de habitaciones por país'.",
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
