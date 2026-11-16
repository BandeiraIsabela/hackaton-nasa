"""RotaDF - planting guide for farmers in Brazil's Federal District.
Click the map -> climate, terrain and season -> what to plant now and next (rotation).
Run: earthengine authenticate ; streamlit run app.py   (.env: GEE_PROJECT=your-project)
"""
from dotenv import load_dotenv
load_dotenv()
import os, json, math, inspect, tempfile, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from time import monotonic, time
import ee, folium, pandas as pd, requests, streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium
import aptidao, enso, zarc  # cobertura do solo, El Niño/La Niña (NOAA) e calendário oficial ZARC de Brasília

st.set_page_config(page_title="RotaDF", layout="wide", initial_sidebar_state="collapsed")

EE_PRAZO, EE_PAUSA = 30, 120  # s: desiste do Initialize; não tenta de novo depois de uma falha (#19)
EE_DISCOVERY_IDADE = 24 * 3600  # s: idade máxima do discovery da API do Earth Engine salvo em disco (#22)

def instalar_cache_discovery():
    """O `ee` monta os clientes com `cache_discovery=False` e baixa o discovery da API duas vezes a cada Initialize
    (#22). Guarda o documento em memória e em `$ROTADF_CACHE/ee_discovery.json` por EE_DISCOVERY_IDADE s. O hook é no
    `_retrieve_discovery_doc` (privado) do googleapiclient: se ele sumir ou mudar de assinatura, segue sem cache.
    Qualquer erro de leitura cai no download normal; erro de gravação é ignorado."""
    from googleapiclient import discovery
    original = getattr(discovery, "_retrieve_discovery_doc", None)
    if original is None or getattr(original, "rotadf", False): return  # o script roda de novo a cada rerun
    try:
        if list(inspect.signature(original).parameters)[:2] != ["url", "http"]: return
    except (TypeError, ValueError): return
    trava, memoria = threading.Lock(), {}

    def retrieve(url, http, *a, **kw):
        if "earthengine.googleapis.com" not in url or "$discovery" not in url:
            return original(url, http, *a, **kw)
        arquivo = os.path.join(os.environ.get("ROTADF_CACHE") or os.path.expanduser("~/.cache/rotadf"), "ee_discovery.json")
        with trava:  # duas tentativas ao mesmo tempo: a segunda espera e lê o que a primeira baixou
            if url in memoria and time() - memoria[url][0] < EE_DISCOVERY_IDADE:
                return memoria[url][1]
            try:
                quando = os.path.getmtime(arquivo)
                with open(arquivo, encoding="utf-8") as f: salvo = json.load(f)
                if salvo["url"] == url and time() - quando < EE_DISCOVERY_IDADE:
                    json.loads(salvo["doc"])
                    memoria[url] = (quando, salvo["doc"])
                    return salvo["doc"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
            doc = original(url, http, *a, **kw)
            memoria[url] = (time(), doc)
            try:
                os.makedirs(os.path.dirname(arquivo), exist_ok=True)
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(arquivo), delete=False) as f:
                    json.dump({"url": url, "doc": doc}, f)
                os.replace(f.name, arquivo)  # atômico: quem lê nunca vê um arquivo pela metade
            except OSError:
                pass
            return doc

    retrieve.rotadf = True
    discovery._retrieve_discovery_doc = retrieve
instalar_cache_discovery()

@st.cache_resource(show_spinner=False)
def _estado_ee():
    """Um por processo: o `cache_resource` não guarda exceções, então a falha fica aqui e não refaz o init a cada rerun."""
    return {"ok": False, "erro": "", "falhou": None, "tentativa": None, "inicio": None, "registrada": None,
            "desistiu": None, "trava": threading.RLock()}

def _registrar_ee(e, fut):
    """Fim de uma tentativa, uma vez só. Roda na thread do Initialize (callback) ou na do script: nada de st.* aqui."""
    with e["trava"]:
        if e["registrada"] is fut or not fut.done(): return
        e["registrada"] = fut
        erro = fut.exception()
        if erro is None: e.update(ok=True, erro="", falhou=None)
        else: e.update(ok=False, erro=str(erro)[:300], falhou=monotonic())

def _ee_pendente(e):
    fut = e["tentativa"]
    return fut is not None and not fut.done() and e["desistiu"] is not fut

def iniciar_ee():
    """Começa o Initialize em segundo plano, sem esperar (#20): a página desenha enquanto ele roda. Uma tentativa por
    vez no processo; depois de uma falha, só tenta de novo após EE_PAUSA s (#19). Uma tentativa ainda presa não é
    repetida."""
    e = _estado_ee()
    with e["trava"]:
        if e["ok"] or (e["falhou"] is not None and monotonic() - e["falhou"] < EE_PAUSA): return
        if e["tentativa"] is not None and not e["tentativa"].done(): return
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(ee.Initialize, project=os.environ.get("GEE_PROJECT"))
        ex.shutdown(wait=False)
        e.update(tentativa=fut, inicio=monotonic())
        fut.add_done_callback(lambda f: _registrar_ee(e, f))  # RLock: se já terminou, roda aqui mesmo

def estado_ee(esperar=True):
    """(ok, erro) do Earth Engine, chamado só onde um dado dele é necessário. Com a tentativa em andamento, espera por
    ela até EE_PRAZO s desde o início (o `setDeadline` não vale antes do Initialize e é por requisição, multiplicado
    pelas retentativas); com `esperar=False` devolve (None, "") nesse caso."""
    iniciar_ee()  # reruns só de fragmento também tentam de novo depois da pausa
    e = _estado_ee()
    fut = e["tentativa"]
    if _ee_pendente(e):
        if not esperar: return None, ""
        try:
            with st.spinner("Connecting to Earth Engine…"):
                fut.result(timeout=max(0, EE_PRAZO - (monotonic() - e["inicio"])))
        except FuturesTimeout:
            with e["trava"]:
                if not fut.done():
                    e.update(desistiu=fut, ok=False, erro=f"Earth Engine did not respond within {EE_PRAZO} s", falhou=monotonic())
        except Exception:
            pass  # o erro é registrado abaixo
    if fut is not None: _registrar_ee(e, fut)  # o callback pode ainda não ter rodado quando `result` volta
    return e["ok"], e["erro"]

iniciar_ee()

TODAY = dt.date.today()
REFERENCIA = TODAY.replace(day=1).isoformat()  # cache mensal; janela de 12 meses termina no mês passado
NDVI_VIS = {"min": 0, "max": 0.9, "palette": ["#8c510a", "#d8b365", "#f6e8c3", "#a6d96a", "#1a9850", "#006837"]}
SAR_VIS = {"min": -25, "max": -5}
# Limites do mapa: DF (lat -16.05/-15.50, lon -48.29/-47.31) + ~50 km de margem
DF_BOUNDS = dict(min_lat=-16.50, max_lat=-15.05, min_lon=-48.76, max_lon=-46.84)
MES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DIAS = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def lee(img, size=2, enl=4.4):
    """Filtro Lee adaptativo (speckle) em escala linear; entrada e saída em dB."""
    lin = ee.Image(10).pow(img.divide(10))
    k = ee.Kernel.square(size, "pixels")
    m = lin.reduceNeighborhood(ee.Reducer.mean(), k)
    v = lin.reduceNeighborhood(ee.Reducer.variance(), k)
    ci2 = v.divide(m.pow(2))
    w = ci2.subtract(1 / enl).divide(ci2).max(0)
    return m.add(w.multiply(lin.subtract(m))).log10().multiply(10)

def s1(geom):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select("VH"))

def month_series(build, band, geom, scale, inicio, n=12):
    def f(i):
        s = ee.Date(inicio).advance(i, "month")
        v = build(s, s.advance(1, "month")).reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e9).get(band, -9999)
        return ee.Feature(None, {"m": s.format("YYYY-MM"), "v": v})
    feats = ee.FeatureCollection(ee.List.sequence(0, n - 1).map(f)).getInfo()["features"]
    ser = pd.Series({x["properties"]["m"]: x["properties"].get("v") for x in feats}, dtype="float")
    return ser.where(ser > -9000)  # meses sem dado (ex.: IMERG ainda não publicado) viram NaN

# ---------------- Culturas (referência agronômica geral, a validar com Emater-DF / Embrapa) ----------------
K = "nome emoji fam meses tmin tmax agua irrig dif fixa_n obj ciclo ha_min cobre desc".split()
_C = [
 ("Soybean","🌱","Fabaceae",{10,11,12},20,30,"alta",0,3,1,{"renda"},4,20,0,"Planted in the rainy season; needs machinery and inputs."),
 ("Corn","🌽","Poaceae",{10,11,12,1,2},18,30,"média",0,2,0,{"renda","comida"},4,3,0,"Grain or fresh corn; good to sell and to feed the family."),
 ("Beans","🫘","Fabaceae",{10,11,2,3,5,6,7},15,27,"média",0,2,1,{"comida","renda"},3,0.5,0,"Short cycle; adds nitrogen to the soil."),
 ("Sorghum","🌾","Poaceae",{11,12,1,2,3},20,32,"baixa",0,1,0,{"renda","terra"},4,5,0,"Drought-tolerant; good after soybean or corn."),
 ("Cassava","🥔","Euphorbiaceae",{9,10,11,12,4,5},18,30,"baixa",0,1,0,{"comida","renda"},8,0.3,0,"Hardy and easy; tolerates little rain."),
 ("Sweet potato","🍠","Convolvulaceae",{10,11,12,1,2,3},20,30,"baixa",0,1,0,{"comida"},4,0.2,1,"Easy, covers the ground and needs little water."),
 ("Squash","🎃","Cucurbitaceae",{9,10,11,12,1,2},20,30,"média",0,1,0,{"comida","renda"},4,0.3,1,"Easy to grow and covers the soil well."),
 ("Green manure (crotalaria/millet)","🌿","Adubo verde",{10,11,12,1,2,3,4},20,32,"baixa",0,1,1,{"terra"},3,0,1,"No harvest, but it protects and fertilizes the soil."),
 ("Pasture grass (brachiaria)","🐄","Poaceae",{10,11,12,1,2,3},20,32,"baixa",0,1,0,{"terra","renda"},24,2,1,"Covers the soil and feeds cattle as pasture."),
 ("Lettuce","🥬","Asteraceae",{3,4,5,6,7,8,9},14,24,"alta",1,2,0,{"renda","comida"},2,0.05,0,"Quick harvest; dislikes heat and needs irrigation."),
 ("Collard greens","🥬","Brassicaceae",{2,3,4,5,6,7,8,9},15,26,"média",0,1,0,{"comida","renda"},3,0.05,0,"Hardy, with a long harvest period."),
 ("Fresh herbs (parsley and chives)","🌿","Cheiro-verde",{3,4,5,6,7,8,9},14,26,"média",1,1,0,{"renda","comida"},3,0.05,0,"Needs little space; sells well at markets."),
 ("Tomato","🍅","Solanaceae",{3,4,5,6},18,28,"alta",1,3,0,{"renda"},5,0.2,0,"Profitable, but labor-intensive and disease-prone."),
 ("Carrot","🥕","Apiaceae",{3,4,5,6,7,8},15,25,"média",1,2,0,{"renda","comida"},4,0.2,0,"Prefers cooler weather and loose soil."),
 ("Strawberry","🍓","Rosaceae",{4,5,6},14,25,"alta",1,3,0,{"renda"},8,0.1,0,"High value; needs care and irrigation."),
 ("Coffee","☕","Rubiaceae",{11,12,1,2},18,25,"média",0,3,0,{"renda"},36,2,1,"Perennial crop; takes years to produce."),
 ("Guava","🍈","Myrtaceae",{11,12,1,2,3},18,30,"média",0,2,0,{"renda","comida"},60,0.5,1,"Perennial fruit tree, traditional in the DF."),
]
CAT = [dict(zip(K, r)) for r in _C]
POR_NOME = {c["nome"]: c for c in CAT}

NIVEIS = {"Never farmed": 1, "I already farm": 2, "Experienced": 3}
AGUA = {"Rain only": False, "I can irrigate": True}
OBJ = {"Income": "renda", "Food": "comida", "Care for the land": "terra"}
TAM = {"Small (about 2 hectares)": 2, "Medium (about 10 hectares)": 10,
       "Large (about 50 hectares)": 50, "Very large (about 200 hectares)": 200}
NUNCA, PARADA = "Never planted / don't know", "Idle land (nothing)"
SOLO_PADRAO = next(iter(zarc.SOLOS))  # "Don't know (use medium soil)"
DF_LIMITES = (-16.05, -15.50, -48.29, -47.31)  # o ZARC usado é o do município de Brasília, que cobre o DF inteiro

# ---------------- Geografia ----------------
def pontos(gj): return json.loads(gj)["coordinates"][0]

def centro_de(gj):
    p = pontos(gj)[:-1]
    return sum(x[1] for x in p) / len(p), sum(x[0] for x in p) / len(p)

def area_ha(gj):
    p = pontos(gj); c = math.cos(math.radians(centro_de(gj)[0]))
    xy = [(x[0] * 111320 * c, x[1] * 110540) for x in p]
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(xy, xy[1:]))) / 2 / 1e4

def quadrado(lat, lon, ha):
    meio = math.sqrt(ha * 1e4) / 2; dy = meio / 110540; dx = meio / (111320 * math.cos(math.radians(lat)))
    return json.dumps({"type": "Polygon", "coordinates": [[[lon - dx, lat - dy], [lon + dx, lat - dy],
                       [lon + dx, lat + dy], [lon - dx, lat + dy], [lon - dx, lat - dy]]]})

# ---------------- Dados: NASA POWER (clima) e Earth Engine (terreno, umidade do solo, vegetação, radar) ----------------
# Grades das fontes regionais: (borda sul/oeste da primeira célula, tamanho da célula), em graus.
# NASA POWER (meteorologia, MERRA-2): células de 0,5° lat × 0,625° lon centradas em lat = 0,5·k e lon = -180 + 0,625·k
# (conferido na API: -15,76 e -15,74 caem em células diferentes, e -47,83 e -47,80 também).
# FLDAS (Earth Engine): pixels de 0,1° com bordas a partir de -180 / -60 (bbox do catálogo), centro em x,x5.
GRADE_POWER = ((-90.25, 0.5), (-180.3125, 0.625))
GRADE_FLDAS = ((-60.0, 0.1), (-180.0, 0.1))

def celula(lat, lon, grade):
    """Centro (lat, lon) da célula da grade onde fica o ponto. Um ponto na divisa vai sempre para a célula
    ao norte/leste (floor), então o mesmo ponto dá sempre a mesma célula."""
    def c(x, borda, passo):
        return round(borda + (math.floor(round((x - borda) / passo, 9)) + 0.5) * passo, 6)
    (b_lat, p_lat), (b_lon, p_lon) = grade
    return float(c(lat, b_lat, p_lat)), float(c(lon, b_lon, p_lon))

class _NaoEstaNoCache(Exception):
    """Levantada por `_so_cache`: a fonte foi chamada só para ler o cache."""

def _so_cache(): raise _NaoEstaNoCache

def ler_clima_df(hoje=None):
    """Lê e valida a cada execução, sem esconder mudanças, ausência ou corrupção no cache."""
    hoje = hoje or dt.date.today()
    try:
        with open(os.path.join(os.path.dirname(__file__), "data", "clima_power_df.json"), encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        periodo = dados["periodo"]
        if (type(dados["versao"]) is not int or dados["versao"] != 1 or dados["fonte"] != "NASA POWER"
                or type(periodo["inicio"]) is not int or periodo["inicio"] != 2015
                or type(periodo["fim"]) is not int or not 2015 <= periodo["fim"] < hoje.year
                or dados["grade"] != {"origem_lat": GRADE_POWER[0][0], "passo_lat": GRADE_POWER[0][1],
                                     "origem_lon": GRADE_POWER[1][0], "passo_lon": GRADE_POWER[1][1]}
                or not isinstance(dados["celulas"], dict) or not dados["celulas"]):
            return None
        celulas = {}
        for chave, meses in dados["celulas"].items():
            coords = tuple(float(x) for x in chave.split(","))
            if (len(coords) != 2 or not all(math.isfinite(x) for x in coords)
                    or celula(*coords, GRADE_POWER) != coords or f"{coords[0]:g},{coords[1]:g}" != chave
                    or set(meses) != {str(m) for m in range(1, 13)}):
                return None
            for valores in meses.values():
                if (not isinstance(valores, dict) or set(valores) != {"t", "chuva", "ur"}
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in valores.values())):
                    return None
            celulas[chave] = {int(m): valores for m, valores in meses.items()}
        return {"periodo": periodo, "celulas": celulas}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None

@st.cache_data(show_spinner=False, persist="disk")
def clima_power(lat, lon, ano_fim, _calc=None):
    """Clima da célula da POWER com centro (lat, lon), média mensal de 2015 a `ano_fim`. O cache é por célula e
    fica em disco (sobrevive a um reinício; falha não é gravada porque a exceção sobe). Sem ttl, que o Streamlit
    ignora com persist="disk": `ano_fim` está na chave, então na virada do ano a consulta é outra. `_calc` fica fora
    da chave do cache (começa com _) e serve para `coletar` ler ou preencher o cache na thread principal."""
    if _calc: return _calc()
    r = requests.get("https://power.larc.nasa.gov/api/temporal/monthly/point", timeout=40, params=dict(
        parameters="T2M,PRECTOTCORR,RH2M", community="AG", longitude=lon, latitude=lat,
        start=2015, end=ano_fim, format="JSON"))
    r.raise_for_status(); p = r.json()["properties"]["parameter"]
    def med(par, m):
        v = [x for k, x in p[par].items() if len(k) == 6 and k.endswith(f"{m:02d}") and x > -900]
        return sum(v) / len(v) if v else None
    out = {}
    for m in range(1, 13):
        ch = med("PRECTOTCORR", m)
        out[m] = dict(t=med("T2M", m), chuva=None if ch is None else ch * DIAS[m - 1], ur=med("RH2M", m))
    return out

@st.cache_data(show_spinner=False)
def terreno(gj, _calc=None):
    if _calc: return _calc()
    geom = ee.Geometry(json.loads(gj))
    e = ee.Image("USGS/SRTMGL1_003")
    r = e.addBands(ee.Terrain.slope(e)).reduceRegion(ee.Reducer.mean(), geom, 30, maxPixels=1e9).getInfo()
    return dict(alt=r["elevation"], decl=r["slope"])

@st.cache_data(show_spinner=False)
def umidade_solo(lat, lon, referencia, _calc=None):
    """Umidade do solo (0-10 cm, NASA FLDAS, mensal, 0,1° ≈ 11 km) no pixel com centro (lat, lon): último mês
    publicado vs. o mesmo mês de 2010 até o ano anterior ao dele. A série inteira vem num único getInfo e o percentil
    é calculado aqui. O cache é por pixel e mês de referência; `_calc` como em `clima_power`."""
    if _calc: return _calc()
    ponto, banda = ee.Geometry({"type": "Point", "coordinates": [lon, lat]}), "SoilMoi00_10cm_tavg"
    ref = dt.date.fromisoformat(referencia)
    fim = dt.date(ref.year + (ref.month == 12), ref.month % 12 + 1, 1)  # inclui o mês de referência, se publicado
    col = (ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001").select(banda)
           .filterDate("2010-01-01", fim.isoformat()))
    feats = ee.FeatureCollection(col.map(lambda img: ee.Feature(None, {
        "m": img.date().format("YYYY-MM"),
        "v": img.reduceRegion(ee.Reducer.mean(), ponto, 11132).get(banda)}))).getInfo()["features"]
    serie = {}  # (ano, mês) -> valor; meses sem dado (pixel mascarado, ex.: água) ficam de fora
    for x in feats:
        m, v = x["properties"].get("m"), x["properties"].get("v")
        if m and v is not None and v == v and v > -9000:
            serie[(int(m[:4]), int(m[5:7]))] = float(v)
    if not serie: return None
    ano, mes = max(serie)
    atual = serie[(ano, mes)]
    vals = [v for (a, m), v in serie.items() if m == mes and 2010 <= a < ano]  # anos sem dado no mês são ignorados
    return dict(pct=sum(v < atual for v in vals) / len(vals), mes=mes, ano=ano) if vals else None

def _ou_none(ser): return None if ser.dropna().empty else ser

@st.cache_data(show_spinner=False)
def ndvi(gj, referencia, _calc=None):
    if _calc: return _calc()
    geom = ee.Geometry(json.loads(gj))
    ref = dt.date.fromisoformat(referencia)
    inicio = ref.replace(year=ref.year - 1).isoformat()
    return _ou_none(month_series(lambda s, e: ee.ImageCollection("MODIS/061/MOD13Q1")
                    .filterDate(s, e).select("NDVI").mean(), "NDVI", geom, 250, inicio) * 1e-4)

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def enso_oni(referencia, _calc=None):
    """Fase do El Niño / La Niña pelo índice ONI da NOAA (vale para o planeta todo, não depende da terra).
    `referencia` só renova a chave na virada do mês; o ttl de 6 h pega a atualização mensal da NOAA."""
    if _calc: return _calc()
    return enso.classificar(enso.baixar_oni())

@st.cache_data(show_spinner=False)
def cobertura(gj, referencia, _calc=None):
    """O que existe na terra (Dynamic World, JRC Global Surface Water, WDPA), para dizer se dá para plantar."""
    if _calc: return _calc()
    return aptidao.calcular(ee.Geometry(json.loads(gj)), referencia)

def tentar(fontes):
    """Sucessos (inclusive None) ficam no cache de cada fonte. Falhas ficam só nesta sessão por 120s desde a
    conclusão, por fonte e argumentos (célula para clima/solo). Apenas misses sem pausa rodam em paralelo.
    Todas as leituras/escritas st.* ficam aqui, na thread principal; workers chamam somente __wrapped__."""
    falhas = st.session_state.setdefault("falhas_fontes", {})
    agora = monotonic()
    for chave, (quando, erro) in list(falhas.items()):
        if agora - quando >= 120:
            del falhas[chave]
    d, erros, tarefas = {}, {}, {}
    for k, (fn, args) in fontes.items():
        try:
            d[k] = fn(*args, _calc=_so_cache)
            falhas.pop((k, args), None)  # outra sessão pode já ter recuperado esta fonte
        except _NaoEstaNoCache:
            falha = falhas.get((k, args))
            if falha:
                d[k], erros[k] = None, falha[1]
            else:
                tarefas[k] = (fn, args)
    if tarefas:
        with ThreadPoolExecutor(max_workers=len(tarefas)) as ex:
            futuros = {k: ex.submit(fn.__wrapped__, *args) for k, (fn, args) in tarefas.items()}
        for k, futuro in futuros.items():
            fn, args = tarefas[k]
            try: d[k] = futuro.result()
            except Exception as e:
                d[k] = None
                erros[k] = str(e)[:300]
                falhas[(k, args)] = (monotonic(), erros[k])
            else:
                fn(*args, _calc=lambda valor=d[k]: valor)
                falhas.pop((k, args), None)
    return d, erros

def coletar(gj, referencia, clima_arquivo):
    """Junta tudo o que o satélite sabe da área. O que falhar fica None (e a tela avisa que não temos o dado).
    Não cacheia resultados agregados: `tentar` lê sucessos por fonte e controla o intervalo das falhas.
    Clima local precede a API; clima/solo usam sua grade, terreno a geometria e séries a referência mensal."""
    lat, lon = centro_de(gj)
    ref = dt.date.fromisoformat(referencia)
    grade_clima = celula(lat, lon, GRADE_POWER)
    chave_clima = f"{grade_clima[0]:g},{grade_clima[1]:g}"
    local = clima_arquivo and clima_arquivo["celulas"].get(chave_clima)
    periodo = clima_arquivo["periodo"] if local else {"inicio": 2015, "fim": ref.year - 1}
    # Sem cache agregado, o arquivo validado mais recente é usado mesmo após erros ou cache antigo da API.
    fontes = {} if local else {"clima": (clima_power, (*grade_clima, ref.year - 1))}
    fontes["enso"] = (enso_oni, (referencia,))  # NOAA, não depende do Earth Engine
    d, erros = {}, {}
    if fontes and estado_ee(esperar=False)[0] is None:  # ainda conectando: o clima não espera pelo Earth Engine (#20)
        d, erros = tentar(fontes)
        fontes = {}
    ee_ok, ee_erro = estado_ee()
    if ee_ok:
        fontes["terreno"] = (terreno, (gj,))
        fontes["solo"] = (umidade_solo, (*celula(lat, lon, GRADE_FLDAS), referencia))
        fontes["veg"] = (ndvi, (gj, referencia))
        fontes["cobertura"] = (cobertura, (gj, referencia))
    d2, erros2 = tentar(fontes)
    d.update(d2); erros.update(erros2)
    if local: d["clima"] = local
    d = {k: d[k] for k in ("clima", "terreno", "solo", "veg", "enso", "cobertura") if k in d}  # mesma ordem de antes
    if not ee_ok:
        erros["gee"] = ee_erro
    d["erros"] = erros
    d["clima_periodo"] = periodo
    return d

@st.cache_data(show_spinner=False)
def radar_vh(gj, referencia, _calc=None):
    """Série mensal do radar Sentinel-1 VH com filtro Lee (só no modo técnico; None se não houver imagem)."""
    if _calc: return _calc()
    geom = ee.Geometry(json.loads(gj))
    ref = dt.date.fromisoformat(referencia)
    inicio = ref.replace(year=ref.year - 1).isoformat()
    return _ou_none(month_series(lambda s, e: lee(s1(geom).filterDate(s, e).median()).rename("VH"), "VH", geom, 30, inicio))

# ---------------- Recomendação (regras explicáveis) ----------------
def janela(mes, n): return [(mes - 1 + i) % 12 + 1 for i in range(max(1, min(n, 4)))]

def clima_janela(clima, mes, n=4):
    if not clima: return None
    j = [clima[m] for m in janela(mes, n)]
    if any(x["t"] is None or x["chuva"] is None for x in j): return None
    ur = [x["ur"] for x in j]
    return dict(t=sum(x["t"] for x in j) / len(j), chuva=sum(x["chuva"] for x in j),
                ur=None if None in ur else sum(ur) / len(ur))

def avaliar(c, d, p, mes, ultimo):
    if mes not in c["meses"]: return "fora"
    if c["irrig"] and not p["irriga"]: return "irrigar"
    pts, bom, cuid = 50, [], []
    cj = clima_janela(d.get("clima"), mes, c["ciclo"])
    if cj:
        if c["tmin"] - 1 <= cj["t"] <= c["tmax"] + 1:
            pts += 15; bom.append(f"The expected temperature ({cj['t']:.0f} °C on average) suits it.")
        else:
            pts -= 25; cuid.append(f"It will likely be too {'hot' if cj['t'] > c['tmax'] else 'cold'} for it ({cj['t']:.0f} °C on average).")
        if not p["irriga"]:
            if cj["chuva"] >= {"baixa": 150, "média": 300, "alta": 450}[c["agua"]]:
                pts += 10; bom.append(f"Enough rain is expected ({cj['chuva']:.0f} mm) without irrigation.")
            else:
                pts -= 40; cuid.append(f"Too little rain is expected ({cj['chuva']:.0f} mm) for it without irrigation.")
    else:
        cuid.append("No climate data right now: we could not check whether the weather suits it.")
    if c["dif"] > p["nivel"]:
        pts -= 20 * (c["dif"] - p["nivel"]); cuid.append("Needs more experience: consider asking Emater-DF for support.")
    elif p["nivel"] == 1 and c["dif"] == 1:
        pts += 8; bom.append("A good choice for beginners.")
    if p["ha"] < c["ha_min"]:
        pts -= 25; cuid.append(f"Usually pays off on areas of {c['ha_min']:g} hectares or more.")
    if p["obj"] in c["obj"]:
        pts += 12; bom.append({"renda": "Sells well.", "comida": "Good for feeding the family.",
                               "terra": "Helps care for the land."}[p["obj"]])
    if ultimo:
        if ultimo["fam"] == c["fam"]:
            pts -= 35; cuid.append(f"Same plant family as the previous crop ({ultimo['nome']}): pests and diseases may build up.")
        elif c["fixa_n"] and not ultimo["fixa_n"]:
            pts += 10; bom.append("Returns nitrogen to the soil after the previous crop.")
        elif ultimo["fixa_n"] and c["fam"] == "Poaceae":
            pts += 8; bom.append(f"Uses the nitrogen that {ultimo['nome']} leaves in the soil.")
    t = d.get("terreno")
    if t and t["decl"] > 6:
        if c["cobre"]: pts += 10; bom.append("Protects the soil on sloped land.")
        else: pts -= 10; cuid.append("The land is sloped: rain can wash the soil away. Plant along contour lines.")
    veg = d.get("veg"); sem = int((veg < 0.3).sum()) if veg is not None else 0
    if sem >= 4 and c["cobre"]:
        pts += 6; bom.append(f"Covers the land, which had little vegetation for {sem} months last year.")
    s = d.get("solo")
    if s and not p["irriga"] and s["pct"] < 0.3:
        if c["agua"] == "baixa": pts += 8; bom.append("Copes well with the drier-than-normal soil in the region.")
        elif c["agua"] == "alta": pts -= 8; cuid.append("The soil in the region is drier than normal.")
    en = d.get("enso")
    if en and en["fase"] == "El Niño" and en["intensidade"] in ("moderate", "strong", "very strong") and not p["irriga"]:
        if c["agua"] == "alta":
            pts -= 10; cuid.append(f"{cap(enso.rotulo(en))} year: rain in the DF tends to be irregular, with dry spells.")
        elif c["agua"] == "baixa":
            pts += 5; bom.append(f"Copes with the irregular rain of a {enso.rotulo(en)} year.")
    z = zarc.risco_mes(c["nome"], p.get("solo", SOLO_PADRAO), mes, p["irriga"])
    if z:  # ZARC só restringe: fora do calendário ou acima do risco aceito no ano tira pontos; dentro, só explica
        (r, dec), tol = z, zarc.tolerancia(en)
        if r == 0:
            pts -= 30; cuid.append(f"Outside Brasília’s official planting calendar (ZARC) in {MES[mes - 1]}: "
                                   "higher risk of loss and no crop insurance or Proagro.")
        elif r > tol:
            pts -= 10; cuid.append(f"Under ZARC, planting in {MES[mes - 1]} has a {r}% risk of loss "
                                   f"(this year we accept up to {tol}%).")
        else:
            bom.append(f"Within the official calendar (ZARC): {r}% risk when planted on {zarc.nome_dec(dec)}.")
    # O teto serve à barra visual; ordenar por ele apagaria diferenças ambientais acima de 100.
    return dict(c, adequacao=pts, pts=max(0, min(100, round(pts))), bom=bom, cuid=cuid)

NOTA_MIN = 50

def ranking(d, p, mes, ultimo):
    rs = [avaliar(c, d, p, mes, ultimo) for c in CAT]
    ok = [r for r in rs if isinstance(r, dict) and r["pts"] >= NOTA_MIN]  # abaixo disso não vale recomendar
    return sorted(ok, key=lambda r: -r["adequacao"]), [r for r in rs if isinstance(r, str)]

def plano(d, p, mes, ultimo, passos=3):
    """Sequência de plantios (rotação): cada passo escolhe a melhor cultura depois do ciclo da anterior."""
    seq, m = [], mes
    for _ in range(passos):
        for espera in range(7):
            ok, _ = ranking(d, p, (m - 1 + espera) % 12 + 1, ultimo)
            if ok: break
        else:
            break
        top, mm = ok[0], (m - 1 + espera) % 12 + 1
        seq.append((mm, top, espera))
        if top["ciclo"] >= 12: break  # cultura de vários anos: fica no lugar
        m, ultimo = (mm - 1 + top["ciclo"]) % 12 + 1, top
    return seq

def confianca(d):
    if d.get("clima") is None: return "baixa"
    n = sum(d.get(k) is not None for k in ("clima", "terreno", "solo", "veg"))
    return "alta" if n == 4 else "média" if n >= 2 else "baixa"

CAMADAS = ("Vegetation health (green = healthy)", "Radar (raw, noisy)", "Radar (filtered)")

@st.cache_data(show_spinner=False, ttl=3600)
def layers(gj, referencia, _hoje, _calc=None):
    """URLs das camadas do modo técnico: {"urls": {nome: url}, "falhas": {nome: erro}}. Cada camada é montada e
    pedida (getMapId) na sua própria thread, sem nenhuma chamada st.*; a falha de uma não apaga as outras. Nunca
    levanta exceção, então o resultado (com as falhas) fica no cache e o mapa não fica tentando de novo a cada
    execução. A chave muda na virada do mês. Ao renovar pelo ttl, a janela diária termina no dia da consulta.
    `_calc` como em `clima_power` (com `_so_cache`, só lê o cache)."""
    if _calc: return _calc()
    def montar(nome):
        geom = ee.Geometry(json.loads(gj)); box = geom.buffer(1500).bounds()
        end = ee.Date(_hoje)
        if nome == CAMADAS[0]:
            img = (ee.ImageCollection("NASA/HLS/HLSL30/v002").filterBounds(geom).filterDate(end.advance(-120, "day"), end)
                   .map(lambda i: i.updateMask(i.select("Fmask").bitwiseAnd(14).eq(0)).normalizedDifference(["B5", "B4"]))
                   .median().clip(box))
            vis = NDVI_VIS
        else:
            raw = s1(geom).filterDate(end.advance(-60, "day"), end).median()
            img, vis = (raw if nome == CAMADAS[1] else lee(raw)).clip(box), SAR_VIS
        return img.getMapId(vis)["tile_fetcher"].url_format
    with ThreadPoolExecutor(max_workers=3) as ex:
        futuros = {nome: ex.submit(montar, nome) for nome in CAMADAS}
    urls, falhas = {}, {}
    for nome, f in futuros.items():
        try: urls[nome] = f.result()
        except Exception as e: falhas[nome] = str(e)[:300]
    return dict(urls=urls, falhas=falhas)

CARREGANDO_CAMADAS = "Loading the vegetation and radar layers…"

def aviso_camadas(falhas):
    """Mensagem curta sobre as camadas que não puderam ser carregadas."""
    if len(falhas) == len(CAMADAS): return "Could not load the vegetation and radar layers."
    nomes = [f"“{n}”" for n in CAMADAS if n in falhas]
    return (f"Could not load the {nomes[0]} layer." if len(nomes) == 1
            else f"Could not load the {', '.join(nomes[:-1])} and {nomes[-1]} layers.")

# --- MAPA ---
from branca.element import MacroElement, Template

class _TextoPT(MacroElement):
    """Sets the texts of the drawing tool (must run before Draw)."""
    _template = Template("""{% macro script(this, kwargs) %}
    (function () {
      try {
        var t = L.drawLocal.draw.toolbar, h = L.drawLocal.draw.handlers;
        t.actions.title = 'Cancel drawing'; t.actions.text = 'Cancel';
        t.finish.title = 'Finish drawing'; t.finish.text = 'Finish';
        t.undo.title = 'Delete the last point'; t.undo.text = 'Delete last point';
        t.buttons.polygon = 'Draw your land'; t.buttons.rectangle = 'Draw a rectangle';
        h.polygon.tooltip.start = 'Click the map to start marking your land';
        h.polygon.tooltip.cont = 'Click to keep marking';
        h.polygon.tooltip.end = 'Click the first point to finish';
        h.rectangle.tooltip.start = 'Click and drag to draw the rectangle';
        h.simpleshape.tooltip.end = 'Release the mouse to finish';
      } catch (e) {}
    })();
    {% endmacro %}""")

class _LimitarMapa(MacroElement):
    """Trava o mapa: não arrasta para fora da caixa e não afasta além do zoom mínimo."""
    _template = Template("""{% macro script(this, kwargs) %}
    var _m = {{ this._parent.get_name() }};
    _m.setMaxBounds(L.latLngBounds([{{ this.sul }}, {{ this.oeste }}], [{{ this.norte }}, {{ this.leste }}]));
    _m.options.maxBoundsViscosity = 1.0;
    _m.setMinZoom({{ this.zmin }});
    _m.setMaxZoom(17);
    if (_m.getZoom() < {{ this.zmin }}) { _m.setZoom({{ this.zmin }}); }
    try {
      _m.zoomControl._zoomInButton.title = 'Zoom in';
      _m.zoomControl._zoomOutButton.title = 'Zoom out';
    } catch (e) {}
    {% endmacro %}""")
    def __init__(self, b, zmin):
        super().__init__()
        self.sul, self.oeste, self.norte, self.leste, self.zmin = b["min_lat"], b["min_lon"], b["max_lat"], b["max_lon"], zmin

ZOOM_MIN = 9
COR = "#F5B700"  # amarelo de ipê: contrasta com o satélite e com o mapa

def construir_mapa(gj, desenho):
    # obs.: com tiles=None o folium ignora min_zoom/max_zoom; os limites de zoom são aplicados em _LimitarMapa
    m = folium.Map(location=[-15.78, -47.80], zoom_start=10, tiles=None, max_bounds=True, **DF_BOUNDS)
    folium.TileLayer("OpenStreetMap", name="Map", no_wrap=True, show=False).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Satellite", no_wrap=True).add_to(m)
    if desenho:
        _TextoPT().add_to(m)  # antes do Draw
        Draw(draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False,
                           "polygon": {"allowIntersection": False, "shapeOptions": {"color": COR, "weight": 3},
                                       "drawError": {"color": "#e1e100", "message": "Lines cannot cross."}},
                           "rectangle": {"shapeOptions": {"color": COR, "weight": 3}}},
             edit_options={"edit": False, "remove": False}).add_to(m)
    if gj:
        pts = pontos(gj)
        folium.GeoJson(json.loads(gj), name="Your land", style_function=lambda _: {"color": COR, "weight": 3, "fillOpacity": 0.15}).add_to(m)
        m.fit_bounds([[min(p[1] for p in pts), min(p[0] for p in pts)], [max(p[1] for p in pts), max(p[0] for p in pts)]],
                     padding=(40, 40), max_zoom=15)  # centraliza na terra marcada
    else:
        m.fit_bounds([[-16.07, -48.30], [-15.48, -47.30]])  # começa mostrando o DF inteiro
    _LimitarMapa(DF_BOUNDS, ZOOM_MIN).add_to(m)  # o controle de camadas vai à parte, em `grupos_camadas` (#15)
    return m

class _Visiveis(MacroElement):
    """Depois de trocar as camadas, deixa visíveis só as que o usuário tinha ligado (no começo, só a vegetação).
    O st_folium liga todos os grupos de `feature_group_to_add`; aqui os outros são tirados do mapa. Só conta o que
    o usuário fez no controle de camadas (`_handlingClick`), não a troca dos grupos pelo st_folium."""
    _template = Template("""{% macro script(this, kwargs) %}
    (function () {
      var m = {{ this._parent._parent.get_name() }}, grupos = { {% for g in this.grupos %}{{ g.layer_name|tojson }}: {{ g.get_name() }},{% endfor %} };
      var vis = window._rotadf_visiveis;
      if (!vis) {
        vis = window._rotadf_visiveis = {{ this.inicio|tojson }};
        var anotar = function (e) { if (window.layer_control && window.layer_control._handlingClick) vis[e.name] = e.type === 'overlayadd'; };
        m.on('overlayadd', anotar); m.on('overlayremove', anotar);
      }
      for (var n in grupos) { if (!vis[n]) m.removeLayer(grupos[n]); }
    })();
    {% endmacro %}""")
    def __init__(self, grupos):
        super().__init__()
        # Inclui camadas ainda indisponíveis, para a recuperação manter seu padrão sem sobrescrever escolhas.
        self.grupos, self.inicio = grupos, {nome: nome == CAMADAS[0] for nome in CAMADAS}

def grupos_camadas(urls):
    """Camadas do modo técnico, uma por FeatureGroup, para o `feature_group_to_add` do st_folium (#15): o mapa já
    aberto troca só esses grupos e o controle de camadas, sem recarregar; terra, zoom e posição ficam como estão."""
    grupos = []
    for nome in CAMADAS:
        if nome in urls:
            g = folium.FeatureGroup(name=nome, show=nome == CAMADAS[0])
            folium.TileLayer(tiles=urls[nome], attr="Google Earth Engine", name=nome).add_to(g)
            grupos.append(g)
    if grupos:
        fim = folium.FeatureGroup(name="Visible layers", control=False)  # roda depois de todos os grupos
        _Visiveis(list(grupos)).add_to(fim); grupos.append(fim)
    return grupos
# --- FIM MAPA ---

# ---------------- Interface ----------------
import html as _h
# Fontes servidas pelo próprio app (static/, `enableStaticServing`), sem Google Fonts (#12): só o subconjunto
# latino do Google Fonts (acentos do português), licença SIL OFL em static/OFL-*.txt.
CSS = """
@font-face{font-family:'Bricolage Grotesque';font-style:normal;font-weight:500 700;font-display:swap;src:url('app/static/BricolageGrotesque-latin.woff2') format('woff2');unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:'Public Sans';font-style:normal;font-weight:400 600;font-display:swap;src:url('app/static/PublicSans-latin.woff2') format('woff2');unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
:root{--bg:#F2F4F1;--papel:#fff;--tinta:#16241D;--verde:#1D4D3B;--verde-cl:#DDE8E0;--ipe:#E8B000;--terra:#9C4A2F;--linha:#D9DFDA;--mut:#5F6B64}
.stApp{background:var(--bg);color:var(--tinta)}
.stApp,.stMarkdown,.stMarkdown p,label,button,input,[data-baseweb]{font-family:'Public Sans',system-ui,sans-serif}
.block-container{max-width:1280px;padding:1.2rem 1.5rem 4rem}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
h1,h2,h3,h4,h5,.marca,.passo span,.trilha b{font-family:'Bricolage Grotesque',system-ui,sans-serif;color:var(--verde);letter-spacing:-.015em}
.topo{display:flex;flex-wrap:wrap;align-items:baseline;gap:.3rem 1.2rem;padding:.2rem 0 1rem;border-bottom:1px solid var(--linha);margin-bottom:1.2rem}
.marca{font-size:1.5rem;font-weight:700;display:inline-flex;align-items:center;gap:.5rem}
.marca::before{content:"";width:.8rem;height:.8rem;border-radius:50%;background:var(--ipe)}
.lema{color:var(--mut)}
iframe[title*="folium"]{border:1px solid var(--linha);border-radius:12px;height:clamp(400px,68vh,700px)!important;background:#dfe6e0}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--linha)!important;border-radius:12px;background:var(--papel)}
[data-testid="stExpander"]{border:1px solid var(--linha);border-radius:12px;background:var(--papel)}
[data-testid="stWidgetLabel"] p{font-size:.86rem;color:var(--mut);font-weight:500}
.stButton button{border-radius:8px;border:1px solid var(--verde);color:var(--verde);background:transparent}
.stButton button:hover{background:var(--verde-cl);border-color:var(--verde);color:var(--verde)}
.passo{display:flex;align-items:center;gap:.6rem;margin:.3rem 0 .6rem}
.passo b{width:1.6rem;height:1.6rem;border-radius:50%;background:var(--verde);color:#fff;display:grid;place-items:center;font-size:.85rem}
.passo span{font-weight:700;font-size:1.2rem}
h2.sec{font-size:1.6rem;margin:2.4rem 0 .9rem;font-weight:700}
.destaque{background:var(--verde);color:#fff;border-radius:14px;padding:1.6rem 1.7rem}
.destaque h3{color:#fff;font-size:1.8rem;margin:0 0 .2rem;font-weight:700}
.destaque p{margin:0 0 1rem;color:#CFE0D6}
.destaque small{color:#CFE0D6;display:block;margin-top:.4rem}
.destaque .fit{background:rgba(255,255,255,.22)}
.destaque .fit i{background:var(--ipe)}
.destaque h5{color:#fff;font-size:1.02rem;margin:.2rem 0 .4rem}
.duas{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem 2.2rem;margin-top:1.2rem}
.destaque ul{list-style:none;padding:0;margin:0}
.destaque li{position:relative;padding-left:1.1rem;margin:.4rem 0;font-size:.95rem;line-height:1.45}
.destaque li::before{content:"";position:absolute;left:0;top:.55em;width:.5rem;height:.5rem;border-radius:50%;background:#fff;opacity:.5}
.destaque li.ok::before{background:var(--ipe);opacity:1}
.fit{display:block;height:6px;border-radius:3px;background:var(--verde-cl);overflow:hidden;min-width:90px}
.fit i{display:block;height:100%;background:var(--verde);border-radius:3px}
.linha{display:grid;grid-template-columns:minmax(0,1.3fr) 150px minmax(0,1.6fr);gap:1rem;align-items:center;padding:.95rem .2rem;border-bottom:1px solid var(--linha)}
.linha b{display:block;font-weight:600}
.linha div span{display:block;color:var(--mut);font-size:.88rem}
.linha p{margin:0;font-size:.92rem}
.trilha{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.trilha li{position:relative;padding:1.2rem 1.2rem 0 0}
.trilha li::before{content:"";position:absolute;top:0;left:0;right:0;border-top:2px solid var(--linha)}
.trilha li::after{content:"";position:absolute;top:-5px;left:0;width:10px;height:10px;border-radius:50%;background:var(--ipe)}
.trilha time{display:block;text-transform:capitalize;color:var(--mut);font-size:.88rem}
.trilha b{font-size:1.25rem;display:block;font-weight:700}
.trilha span{display:block;font-size:.9rem;margin-top:.2rem}
.trilha em{display:block;color:var(--terra);font-size:.85rem;font-style:normal;margin-top:.3rem}
.dados{margin:0;border-top:1px solid var(--linha)}
.dados>div{display:grid;grid-template-columns:minmax(150px,1fr) minmax(120px,1fr) minmax(0,1.6fr);gap:.3rem 1rem;padding:.75rem 0;border-bottom:1px solid var(--linha)}
.dados dt{font-weight:600}.dados dd{margin:0}
.dados .fonte{color:var(--mut);font-size:.88rem}
.dados .falta{color:var(--terra);font-weight:600}
.nota{border-left:3px solid var(--ipe);padding:.3rem 1rem;margin:1rem 0;line-height:1.55}
.mut{color:var(--mut);font-size:.92rem}
.apt{border-radius:12px;padding:1rem 1.3rem;margin:.4rem 0 .6rem;border:1px solid var(--linha);background:var(--papel)}
.apt h4{margin:0 0 .3rem;font-size:1.15rem}
.apt ul{margin:.3rem 0 0;padding-left:1.1rem}.apt li{margin:.25rem 0;font-size:.93rem}
.apt.ok{background:var(--verde-cl);border-color:var(--verde-cl)}
.apt.alerta{background:#FFF4D6;border-color:#F1DC9C}
.apt.ruim{background:#F8E3DB;border-color:#E9C2B3}.apt.ruim h4{color:var(--terra)}
@media (max-width:720px){.block-container{padding:.8rem .9rem 3rem}.linha{grid-template-columns:1fr;gap:.4rem}
.dados>div{grid-template-columns:1fr}.destaque{padding:1.2rem}.destaque h3{font-size:1.45rem}h2.sec{font-size:1.35rem}}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

M_CLICK, M_DESENHO = "Click on the map", "Draw outline"
for k, v in dict(map_v=0, centro=None, gj_desenho=None, ultimo_clique=None).items():
    st.session_state.setdefault(k, v)

def limpar():
    st.session_state.update(centro=None, gj_desenho=None, ultimo_clique=None)
    st.session_state["map_v"] += 1  # mapa novo: some a marcação e volta a mostrar o DF

def escolha(rotulo, opcoes, chave, **kw):
    """Seletor em formato de botões (cai para radio em versões antigas do Streamlit)."""
    f = getattr(st, "segmented_control", None)
    if f is None: return st.radio(rotulo, opcoes, key=chave, horizontal=True, **kw)
    return f(rotulo, opcoes, default=opcoes[0], key=chave, **kw) or opcoes[0]

esc = lambda x: _h.escape(str(x))
cap = lambda s: s[:1].upper() + s[1:]
CONF = {"alta": "high", "média": "medium", "baixa": "low"}
html = lambda s: st.markdown(s, unsafe_allow_html=True)
lis = lambda xs, c: "".join(f'<li class="{c}">{esc(x)}</li>' for x in xs)
barra = lambda p: f'<span class="fit"><i style="width:{p}%"></i></span>'

def destaque(r, mes):
    rot = "Great match for your land" if r["pts"] >= 70 else "Good match for your land"
    return ('<section class="destaque">' f'<h3>Plant {esc(r["nome"])} in {MES[mes - 1]}</h3><p>{esc(r["desc"])}</p>'
            f'{barra(r["pts"])}<small>{rot}</small><div class="duas">'
            f'<div><h5>Why it fits</h5><ul>{lis(r["bom"] or ["No strengths found with the data we have."], "ok")}</ul></div>'
            f'<div><h5>Watch out for</h5><ul>{lis(r["cuid"][:4] or ["No warnings with the data we have."], "nd")}</ul></div></div></section>')

def linha(r):
    t = (r["bom"] or r["cuid"] or [r["desc"]])[0]
    return f'<div class="linha"><div><b>{esc(r["nome"])}</b><span>{esc(r["desc"])}</span></div>{barra(r["pts"])}<p>{esc(t)}</p></div>'

def dado(rot, val, fonte):
    v = f"<dd>{esc(val)}</dd>" if val else '<dd class="falta">No data</dd>'
    return f'<div><dt>{rot}</dt>{v}<dd class="fonte">{fonte}</dd></div>'

with st.sidebar:
    st.markdown("### Technical mode")
    tecnico = st.toggle("Vegetation and radar layers on the map")
    st.caption("For remote sensing specialists. The full technical report is at the bottom of the page.")

html('<div class="topo"><span class="marca">RotaDF</span>'
     '<span class="lema">Planting guide for Brazil’s Federal District, built on NASA satellite data</span></div>')
modo = st.session_state.get("modo") or M_CLICK
ha_tam = TAM[st.session_state.get("tam", list(TAM)[1])]
if modo == M_CLICK and st.session_state["centro"]:
    gj = quadrado(*st.session_state["centro"], ha_tam)
else:
    gj = st.session_state["gj_desenho"] if modo == M_DESENHO else None

RENOVAR_CAMADAS = 600  # s; a aba parada confere as camadas a cada 10 min e o `ttl` de `layers` renova a cada hora (#15)

def mapa(gj, tecnico, modo, vez):
    """Mapa, cliques/desenho e aviso das camadas. É um fragmento: com o modo técnico ligado e uma terra marcada, roda
    sozinho a cada RENOVAR_CAMADAS, mesmo com a aba parada, para as URLs das camadas não expirarem (#15). Nessa volta
    sozinho (`vez` já consumido) `layers` é chamado direto: se o cache expirou, renova ali mesmo, e o mapa aberto só
    troca as camadas. Na execução completa, camadas fora do cache ficam para o fim da página (#7)."""
    completa, cam, pendentes = next(vez, False), None, False
    # na execução completa não espera o Earth Engine: o mapa sai antes e as camadas ficam para o fim da página (#20)
    ee_ok = estado_ee(esperar=not completa)[0] if gj and tecnico else None
    if gj and tecnico and ee_ok is not False:
        if completa:
            try: cam = layers(gj, REFERENCIA, TODAY.isoformat(), _calc=_so_cache)
            except _NaoEstaNoCache: pendentes = True
        else:  # dia de hoje, não o da última execução completa (a aba pode estar parada há horas)
            hoje = dt.date.today(); cam = layers(gj, hoje.replace(day=1).isoformat(), hoje.isoformat())
    out = st_folium(construir_mapa(gj, modo == M_DESENHO), height=600, use_container_width=True,
                    key=f"mapa{st.session_state['map_v']}", returned_objects=["last_clicked", "last_active_drawing"],
                    feature_group_to_add=grupos_camadas(cam["urls"] if cam else {}),
                    layer_control=folium.LayerControl(collapsed=True)) or {}
    if modo == M_CLICK:
        c = out.get("last_clicked")
        if c and (c["lat"], c["lng"]) != st.session_state["ultimo_clique"]:
            st.session_state["ultimo_clique"] = (c["lat"], c["lng"]); st.session_state["centro"] = (c["lat"], c["lng"]); st.rerun()
    else:
        dr = out.get("last_active_drawing")
        if dr and json.dumps(dr["geometry"]) != gj:
            st.session_state["gj_desenho"] = json.dumps(dr["geometry"]); st.rerun()
    aviso = st.empty()  # perto do mapa; some quando as camadas entram
    if pendentes: aviso.caption(CARREGANDO_CAMADAS)
    elif cam and cam["falhas"]: aviso.caption(aviso_camadas(cam["falhas"]))
    elif ee_ok is False: aviso.caption(aviso_camadas(CAMADAS))
    return pendentes, aviso

left, right = st.columns([1.7, 1], gap="large")
with left:  # modo técnico desligado: sem run_every, a aba parada não roda nada (nem consulta o Earth Engine)
    renovar = RENOVAR_CAMADAS if gj and tecnico and estado_ee(esperar=False)[0] is not False else None
    camadas_pendentes, aviso = st.fragment(mapa, run_every=renovar)(gj, tecnico, modo, iter([True]))

def carregar_camadas():
    """Calcula as camadas que ficaram fora do cache (agora já no cache de `layers`, por gj) e redesenha o mapa com
    elas. Como a chave é a terra, um clique em outro ponto no meio do caminho não leva estas camadas para a terra nova."""
    if not camadas_pendentes: return
    if not estado_ee()[0]: aviso.caption(aviso_camadas(CAMADAS)); return  # o mapa saiu antes do init terminar (#20)
    try: layers(gj, REFERENCIA, TODAY.isoformat())
    except Exception: aviso.caption(aviso_camadas(CAMADAS)); return  # não deveria acontecer: layers não levanta
    st.rerun()

resultados = st.empty()  # abaixo do mapa e da coluna da direita; o fragmento do passo 2 escreve aqui (#6)

def passo2(gj, ha, area, tecnico, vez):
    """Passo 2 e resultados (#6). É um fragmento: mudar uma resposta roda só ele, e o mapa (outro fragmento) não é
    reconstruído nem mandado de novo ao navegador. Os widgets ficam dentro do fragmento (coluna da direita); os
    resultados não têm widgets e vão para `area`, um `st.empty` criado fora, que cada volta substitui (não acumula).
    O que muda a terra fica fora, no passo 1 e no mapa, e roda a página inteira. A cada execução, data e JSON são
    relidos e `coletar` usa o cache por fonte, permitindo retentativas e viradas sem congelar o agregado.
    Só a primeira execução (completa) pode carregar as camadas adiadas; reruns do formulário não reenviam o mapa."""
    completa = next(vez, False)
    hoje = dt.date.today()
    referencia = hoje.replace(day=1).isoformat()
    html('<div class="passo"><b>2</b><span>Tell us a bit about yourself</span></div>')
    nivel = escolha("Have you farmed before?", list(NIVEIS), "nivel")
    irriga = escolha("Do you have water for irrigation?", list(AGUA), "agua")
    obj = escolha("What is your main goal?", list(OBJ), "obj")
    mes = st.selectbox("When do you plan to plant?", list(range(1, 13)), index=hoje.month - 1, key="mes",
                       format_func=lambda m: MES[m - 1].capitalize() + (" (this month)" if m == hoje.month else ""))
    ult_nome = st.selectbox("What did you last plant on this land?", [NUNCA, PARADA] + [c["nome"] for c in CAT], key="ult")
    solo = st.selectbox("What is your soil like?", list(zarc.SOLOS), key="solo",
                        help="Used for the official planting calendar (ZARC). AD = water available in the soil: "
                             "AD1 holds little water, AD6 holds a lot.")
    with area.container():
        with st.spinner("Checking the satellites… this may take a minute."):
            d = coletar(gj, referencia, ler_clima_df(hoje))
    t = d.get("terreno")
    # sem a cobertura (Earth Engine fora ou falha) não há veredito; a falha aparece no parecer técnico
    apt = aptidao.avaliar(d["cobertura"], t and t["decl"], ha) if d.get("cobertura") is not None else None
    mostrar = not (apt and apt["nivel"] == "inapta") or st.checkbox(
        "Show the recommendation anyway (I know this land can be farmed)", key="forcar")
    with area.container():  # substitui o spinner pelos resultados
        resultado(d, dict(nivel=NIVEIS[nivel], irriga=AGUA[irriga], obj=OBJ[obj], ha=ha, solo=solo), mes,
                  POR_NOME.get(ult_nome), apt, mostrar, gj)
        if completa: carregar_camadas()  # mantém o mapa antes do radar no primeiro carregamento (#7)
        parecer(d, gj, referencia, tecnico)

def bloco_aptidao(apt, ha):
    """Veredito do satélite sobre a área: dá para plantar? (Dynamic World, JRC, WDPA)."""
    if apt is None: return
    mil = lambda v: f"{v:,.0f}"
    motivos = f"<ul>{lis(apt['motivos'], 'nd')}</ul>" if apt["motivos"] else ""
    if apt["nivel"] == "inapta":
        html(f'<div class="apt ruim"><h4>🔴 This area does not look suitable for planting</h4>{motivos}</div>')
    elif apt["nivel"] == "restricao":
        html(f'<div class="apt alerta"><h4>🟡 Can be planted, with care</h4>Estimated plantable area: '
             f'<b>{mil(apt["util_ha"])} ha</b> of {mil(ha)} ha.{motivos}</div>')
    elif apt["nivel"] == "apta":
        html(f'<div class="apt ok"><h4>🟢 Suitable for planting</h4>The satellite found cropland, pasture or grassland on '
             f'{apt["frac_util"]:.0%} of the area, with no significant buildings, water or protected areas.</div>')
    else:
        html(f'<div class="apt">{motivos}</div>')

def dentro_df(gj):
    lat, lon = centro_de(gj)
    return DF_LIMITES[0] <= lat <= DF_LIMITES[1] and DF_LIMITES[2] <= lon <= DF_LIMITES[3]

def resultado(d, perfil, mes, ultimo, apt=None, mostrar=True, gj=None):
    """Aptidão da área, recomendação, plano de rotação e o que sabemos sobre a terra (só elementos sem interação).
    Com a área inapta e `mostrar` falso, pula a recomendação e o plano."""
    ok, motivos = ranking(d, perfil, mes, ultimo)

    if apt is not None:
        html('<h2 class="sec">Can this land be farmed?</h2>')
        bloco_aptidao(apt, perfil["ha"])
        html('<p class="mut">Sources: Dynamic World (Sentinel-2, 10 m, last 6 months), JRC Global Surface Water and WDPA. '
             'The satellite may confuse dry pasture with bare soil, or weedy pasture with native Cerrado: check the map.</p>')
    if gj and not dentro_df(gj):
        html('<p class="nota">Your land seems to be outside the Federal District. The official calendar (ZARC) used here is Brasília’s; '
             'check the one for your municipality.</p>')

    if mostrar:
        recomendacao_e_plano(d, perfil, mes, ultimo, ok)
    else:
        html('<p class="nota">Because this area does not look plantable, we are not showing a planting recommendation. '
             'If you know it can be farmed, tick the option next to the form.</p>')
    sobre_a_terra(d, mes)

def recomendacao_e_plano(d, perfil, mes, ultimo, ok):
    html(f'<h2 class="sec">What suits your land in {MES[mes - 1]}</h2>')
    if not ok:
        html('<p class="nota">Nothing suits your land well at this time of year. ' + (
            "Without irrigation there is not enough rain, and many crops for this season need water. Try another month or see if you can irrigate."
            if not perfil["irriga"] else "Try another month.") + "</p>")
    else:
        html(destaque(ok[0], mes))
        if len(ok) > 1:
            html("<div>" + "".join(linha(r) for r in ok[1:4]) + "</div>")
        if len(ok) > 4:
            html('<p class="mut">Also possible: ' + esc(", ".join(r["nome"] for r in ok[4:7])) + ".</p>")

    html('<h2 class="sec">What next? A rotation plan</h2>')
    seq = plano(d, perfil, mes, ultimo)
    html('<ol class="trilha">' + "".join(
        f'<li><time>{MES[mm - 1]}</time><b>{esc(r["nome"])}</b><span>{esc(r["bom"][0] if r["bom"] else r["desc"])}</span>'
        + (f"<em>The land rests for {e} month(s) first.</em>" if e else "") + "</li>" for mm, r, e in seq) + "</ol>")
    html('<p class="mut">The idea is to alternate plant families and include one that returns nitrogen to the soil (beans, crotalaria), '
         'to care for the land and break pest and disease cycles.</p>')

def sobre_a_terra(d, mes):
    html('<h2 class="sec">What we know about your land</h2>')
    cj, t, s, veg = clima_janela(d.get("clima"), mes, 4), d.get("terreno"), d.get("solo"), d.get("veg")
    en = d.get("enso")
    nasa = "NASA POWER, regional"
    html('<dl class="dados">' + "".join([
        dado("Temperature in the coming months", cj and f"{cj['t']:.0f} °C on average", nasa),
        dado("Expected rain over 4 months", cj and f"{cj['chuva']:.0f} mm", nasa),
        dado("Air humidity", cj and cj["ur"] is not None and f"{cj['ur']:.0f}%", nasa),
        dado("Altitude", t and f"{t['alt']:.0f} m", "NASA SRTM, 30 meters"),
        dado("Terrain", t and ("Flat" if t["decl"] <= 3 else "Sloped" if t["decl"] <= 6 else "Steep"), "NASA SRTM, 30 meters"),
        dado("Soil moisture", s and ("Much drier than normal" if s["pct"] < 0.25 else "Slightly drier than normal"
             if s["pct"] < 0.4 else "Normal" if s["pct"] <= 0.6 else "Wetter than normal")
             + f" ({MES[s['mes'] - 1]} {s['ano']})", "NASA FLDAS, ~11 km, regional"),
        dado("Months with little vegetation last year", veg is not None and f"{int((veg < 0.3).sum())} of 12", "NASA MODIS, 250 meters"),
        dado("Pacific (El Niño / La Niña)", en and f"{cap(enso.rotulo(en))} (ONI {en['oni']:+.1f} °C, {en['trimestre']})",
             "NOAA, ONI index")]) + "</dl>")
    if en:
        tol = zarc.tolerancia(en)
        html(f'<p class="nota"><b>This year’s climate:</b> {esc(en["impacto"])} That is why, in Brasília’s official planting '
             f'calendar (ZARC), we accept dates with up to {tol}% risk of loss.</p>')
    conf = confianca(d)
    html(f'<p class="nota"><b>Confidence: {CONF[conf]}.</b> ' + {"alta": "We got all the data.",
         "média": "Some data is missing; see the items marked No data.",
         "baixa": "Important data, such as climate, is missing. Use this only as a starting point."}[conf] +
         "<br><b>What we don’t know:</b> soil fertility, selling prices, what was planted there before and your resources "
         "(machinery, labor). Climate and soil moisture are regional, not specific to your land. "
         "Talk to an agronomist or Emater-DF before deciding.</p>")

def parecer(d, gj, referencia, tecnico):
    """Parecer e séries são redesenhados junto aos resultados, sem manter erros ou períodos antigos."""
    radar, erros = None, dict(d.get("erros", {}))
    if tecnico and estado_ee()[0]:  # o radar não entra na recomendação: só é consultado no modo técnico
        with st.spinner("Loading the radar…"):
            dados_radar, falhas_radar = tentar({"radar": (radar_vh, (gj, referencia))})
            radar = dados_radar["radar"]
            erros.update(falhas_radar)

    with st.expander("Technical report (for specialists)"):
        periodo = d["clima_periodo"]
        fontes = [("Climate (T2M, rain, RH)", d.get("clima"), f"NASA POWER, monthly mean {periodo['inicio']}–{periodo['fim']}, ~50 km"),
                  ("Elevation and slope", d.get("terreno"), "NASA SRTM, 30 m"),
                  ("Soil moisture 0–10 cm", d.get("solo"), "NASA FLDAS Noah, monthly, 0.1° (~11 km); "
                   "percentile of the latest month vs. the same month from 2010 to the previous year"),
                  ("Monthly NDVI", d.get("veg"), "NASA MODIS MOD13Q1, 250 m (last 12 months)"),
                  ("Radar VH with Lee filter", radar, "Sentinel-1 GRD (ESA), 30 m, Lee 5×5, ENL 4.4"),
                  ("Land cover", d.get("cobertura"), "Dynamic World V1 (Sentinel-2, 10 m, 6-month mode), "
                   "JRC Global Surface Water 1.4 (30 m), WDPA (IUCN Ia–III protected areas)"),
                  ("El Niño / La Niña", d.get("enso"), "NOAA CPC, Oceanic Niño Index (Niño 3.4, 3-month mean)"),
                  ("Planting calendar (ZARC)", zarc.base(), "Brazil’s Ministry of Agriculture (MAPA), ZARC risk table, Brasília "
                   "(data/zarc_brasilia.csv)")]
        disp = lambda n, v: "technical mode only" if n.startswith("Radar") and not tecnico else "yes" if v is not None else "NO"
        st.dataframe(pd.DataFrame([{"Data": n, "Available": disp(n, v), "Source / resolution": f} for n, v, f in fontes]),
                     hide_index=True, width="stretch")
        for k, e in erros.items(): st.caption(f"Failed '{k}': {e}")
        if d.get("clima"):
            st.markdown("**Monthly climate (period mean)**")
            st.dataframe(pd.DataFrame([{"Month": MES[m - 1], "Mean T (°C)": v["t"], "Rain (mm)": v["chuva"], "RH (%)": v["ur"]}
                                       for m, v in d["clima"].items()]).round(1), hide_index=True, width="stretch")
        if d.get("veg") is not None: st.markdown("**Monthly NDVI (MODIS)**"); st.line_chart(d["veg"])
        if radar is not None: st.markdown("**Filtered radar VH (dB)**"); st.line_chart(radar)
        st.markdown("**Method.** Rule-based ranking (not a trained model): planting window, mean temperature and rainfall over the crop cycle "
                    "(up to 4 months) against the crop's range; irrigation needs; difficulty vs. experience; minimum area; goal; "
                    "rotation (same family is penalized; a legume after a non-legume and a grass after a legume earn a bonus); slope > 6° favors "
                    "cover crops; soil moisture below the 30th percentile (without irrigation). NDVI and the Lee filter follow the course notebooks "
                    "NDVI/GEE_multiplas_datas and Lee_filter. Crop ranges and windows are general agronomic references and "
                    "need validation (Emater-DF, Embrapa Hortaliças/Cerrados). Climate and soil data have a coarser resolution (10–50 km) than the land.")
        st.markdown("**ZARC, El Niño and land suitability.** Soybean, corn, beans and sorghum are checked against Brasília's ZARC "
                    "risk table by soil type (AD1–AD6) and by the best 10-day period of the month: outside the zoning costs 30 points, and a risk "
                    "above this year's limit costs 10. ENSO comes from NOAA's ONI (±0.5 °C; confirmed after 5 consecutive 3-month periods); "
                    "with a moderate or stronger El Niño we only accept 20% risk (otherwise 30%) and, without irrigation, crops with high "
                    "water demand lose 10 points and low-demand crops gain 5. Suitability uses the most frequent Dynamic World class "
                    "over 6 months: ≥ 30% buildings, water or wetland, land inside a strictly protected area, or almost only native "
                    "vegetation make the land unsuitable; smaller shares trigger warnings. Heuristic thresholds, to be validated.")

with right, st.container(border=True):
    html('<div class="passo"><b>1</b><span>Mark your land</span></div>')
    escolha("How do you want to mark it?", [M_CLICK, M_DESENHO], "modo", on_change=limpar)
    if modo == M_CLICK:
        st.selectbox("Approximate size", list(TAM), index=1, key="tam", help="1 hectare is roughly the size of a soccer field.")
        st.caption("Click the map in the middle of your land. Wrong spot? Click somewhere else.")
    else:
        st.caption("Use the map's polygon button: click the corners and finish on the first point. "
                   "Made a mistake while drawing? Use Cancel on the map.")
    st.button("Clear and start over", on_click=limpar, disabled=not gj)
    if not gj:
        html('<p class="nota">Mark your land on the map to see what suits it.</p>')
        st.stop()
    ha = area_ha(gj)
    if ha > 5000:
        html('<p class="nota">This area is too large (over 5,000 hectares). Mark only your land.</p>')
        carregar_camadas(); st.stop()
    html(f'<p class="mut">Land marked: about {ha:,.0f} hectares.</p>')

    st.fragment(passo2)(gj, ha, resultados, tecnico, iter([True]))
