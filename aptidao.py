"""Aptidão da área para plantio: o satélite identifica o que existe dentro do polígono.

Fontes (Google Earth Engine):
- Dynamic World V1 (Google/WRI, Sentinel-2, 10 m): classe mais frequente nos últimos 6 meses
  (água, árvores, campo, vegetação alagada, lavoura, arbusto, construção, solo exposto).
- JRC Global Surface Water (Landsat 1984-2021, 30 m): quanto do tempo cada pixel ficou com água.
- WDPA (áreas protegidas): unidades de conservação de proteção integral (IUCN Ia, Ib, II, III).

As regras ficam em avaliar() (sem Earth Engine), para poderem ser testadas e ajustadas.
O `ee` é importado dentro das funções: assim usa o módulo atual de `sys.modules` (nos testes, o falso).
"""

DW_CLASSES = ["agua", "arvores", "campo", "alagado", "lavoura", "arbustos", "construcao", "solo_exposto", "neve"]
NOMES = {"agua": "Water", "arvores": "Trees / forest", "campo": "Pasture / grassland", "alagado": "Wetland / vereda",
         "lavoura": "Cropland", "arbustos": "Cerrado shrubland", "construcao": "Buildings / pavement",
         "solo_exposto": "Bare soil", "neve": "Snow"}
DW_PALETA = ["419bdf", "397d49", "88b053", "7a87c6", "e49635", "dfc35a", "c4281b", "a59b8f", "b39fe1"]
PLANTAVEL = ("campo", "lavoura", "solo_exposto")  # arbustos no DF costumam ser cerrado nativo
UC_RESTRITAS = ["Ia", "Ib", "II", "III"]


def dynamic_world(geom, fim, dias=180):
    import ee
    return (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(geom)
            .filterDate(ee.Date(fim).advance(-dias, "day"), ee.Date(fim)).select("label").mode().rename("label"))  # mode() renomeia para label_mode


def calcular(geom, fim):
    """Faz uma única chamada ao Earth Engine e devolve frações (0-1) de cada cobertura."""
    import ee
    dw =dynamic_world(geom, fim)
    hist = dw.reduceRegion(ee.Reducer.frequencyHistogram(), geom, 10, maxPixels=1e10, bestEffort=True).get("label")
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    agua = (occ.gte(50).rename("perm").addBands(occ.gte(10).And(occ.lt(50)).rename("sazonal"))
            .reduceRegion(ee.Reducer.mean(), geom, 30, maxPixels=1e10, bestEffort=True))
    ucs = (ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(geom)
           .filter(ee.Filter.inList("IUCN_CAT", UC_RESTRITAS)))
    uc_frac = (ee.Image(0).paint(ucs, 1).rename("uc")
               .reduceRegion(ee.Reducer.mean(), geom, 30, maxPixels=1e10, bestEffort=True).get("uc"))
    r = ee.Dictionary({"hist": hist, "agua": agua, "uc_frac": uc_frac,
                       "uc_nomes": ucs.aggregate_array("NAME").distinct()}).getInfo()
    h = {int(float(k)): v for k, v in (r.get("hist") or {}).items()}
    tot = sum(h.values()) or 1
    frac = {c: h.get(i, 0) / tot for i, c in enumerate(DW_CLASSES)}
    return {"cobertura": frac,
            "agua_perm": (r.get("agua") or {}).get("perm") or 0.0,
            "agua_sazonal": (r.get("agua") or {}).get("sazonal") or 0.0,
            "uc_frac": r.get("uc_frac") or 0.0,
            "uc_nomes": r.get("uc_nomes") or [],
            "sem_dado": not h}


def avaliar(info, slope=None, ha=None):
    """Regras de aptidão. Devolve {'nivel': 'apta'|'restricao'|'inapta', 'motivos': [...], 'util_ha': ...}."""
    if info is None or info.get("sem_dado"):
        return {"nivel": "desconhecido", "motivos": ["No cloud-free satellite image of this area in the last 6 months."],
                "util_ha": None, "frac_util": None}
    c = info["cobertura"]
    agua = max(c["agua"], info["agua_perm"])
    bloq, alerta = [], []

    if info["uc_frac"] > 0.05:
        nomes = ", ".join(info["uc_nomes"][:3]) or "protected area"
        bloq.append(f"{info['uc_frac']:.0%} of the area is inside a strictly protected area ({nomes}): "
                    "planting is not allowed.")
    if c["construcao"] >= 0.30:
        bloq.append(f"{c['construcao']:.0%} of the area has buildings, houses or pavement: it looks urban or built up.")
    elif c["construcao"] >= 0.05:
        alerta.append(f"{c['construcao']:.0%} of the area has structures (houses, sheds, roads). They are excluded from the plantable area.")
    if agua >= 0.30:
        bloq.append(f"{agua:.0%} of the area is covered by water (lake, reservoir or river).")
    elif agua >= 0.03:
        alerta.append(f"{agua:.0%} of the area has water. Banks of rivers, lakes and springs are protected (APP, Brazil’s Forest Code): "
                      "do not plant there.")
    if info["agua_sazonal"] >= 0.10:
        alerta.append(f"{info['agua_sazonal']:.0%} of the area has flooded in some years (1984–2021 record): "
                      "risk of waterlogging in the rainy season.")
    if c["alagado"] >= 0.30:
        bloq.append(f"{c['alagado']:.0%} of the area is wetland or vereda: wetlands are protected and the soil stays waterlogged.")
    elif c["alagado"] >= 0.03:
        alerta.append(f"{c['alagado']:.0%} of the area looks like wetland or vereda (a protected area). Avoid planting there.")
    if c["arvores"] >= 0.50:
        alerta.append(f"{c['arvores']:.0%} of the area is covered by trees. If it is native Cerrado forest, clearing it "
                      "requires an environmental permit (Brasília Ambiental/IBRAM) and it may be a Legal Reserve.")
    elif c["arvores"] >= 0.15:
        alerta.append(f"{c['arvores']:.0%} of the area has trees; check whether it is native forest or a Legal Reserve before planting.")
    if slope is not None and slope > 25:
        alerta.append(f"Very steep land (average {slope:.0f}°): use is restricted above 25° and protected (APP) above 45°.")

    if c["arbustos"] >= 0.30:
        alerta.append(f"{c['arbustos']:.0%} of the area looks like native Cerrado (shrubland). Clearing native vegetation requires "
                      "an environmental permit; if it is weedy pasture, ignore this.")

    frac_util = sum(c[k] for k in PLANTAVEL)
    nativa = c["arvores"] + c["arbustos"]
    if frac_util < 0.30 and not bloq:
        if nativa >= 0.50:
            bloq.append(f"{nativa:.0%} of the area is native vegetation (forest or Cerrado), with no cropland or pasture. Planting would "
                        "require clearing it, which needs an environmental permit and must respect the Legal Reserve (20% in the Cerrado).")
        else:
            bloq.append(f"Only {frac_util:.0%} of the area seems to have cropland, pasture or grassland: most of it is not plantable.")
    nivel = "inapta" if bloq else "restricao" if alerta else "apta"
    return {"nivel": nivel, "motivos": bloq + alerta, "frac_util": frac_util,
            "util_ha": None if ha is None else ha * frac_util}
