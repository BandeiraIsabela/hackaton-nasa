"""El Niño / La Niña detection pelo índice ONI (Oceanic Niño Index) da NOAA/CPC.

ONI = média móvel de 3 meses da anomalia de temperatura do mar na região Niño 3.4.
Critério NOAA: El Niño quando ONI >= +0,5 °C e La Niña quando ONI <= -0,5 °C
por pelo menos 5 trimestres sobrepostos seguidos. Com menos que isso, chamamos de "em formação".
"""
import requests

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
LIMIAR = 0.5
MIN_SEQ = 5  # trimestres seguidos para confirmar o evento

# Impactos esperados no DF / Cerrado do Planalto Central. O sinal do ENSO no Centro-Oeste é
# mais fraco que no Sul e no Norte/Nordeste: são tendências, não certezas.
IMPACTO_DF = {
    "El Niño": ("It tends to delay the start of the rainy season on Brazil’s central plateau and make it irregular, with more "
                "dry spells (dry weeks in the middle of the rainy season) and above-normal heat. The end of the rains in "
                "April/May may also be weaker, which works against second-crop corn."),
    "La Niña": ("It tends to bring more regular rain to Brazil’s Center-West in spring and summer, but the rains may end "
                "earlier, with isolated dry spells. In southern Brazil it usually causes drought."),
    "Neutral": "No strong influence from the Pacific: the climate tends to follow the normal regional pattern.",
}


def intensidade(oni):
    a = abs(oni)
    if a >= 2.0: return "very strong"
    if a >= 1.5: return "strong"
    if a >= 1.0: return "moderate"
    if a >= LIMIAR: return "weak"
    return ""


def baixar_oni(timeout=30):
    """Devolve lista [(trimestre, ano, anomalia)] em ordem cronológica."""
    txt = requests.get(ONI_URL, timeout=timeout).text
    out = []
    for ln in txt.splitlines()[1:]:
        p = ln.split()
        if len(p) == 4:
            try:
                out.append((p[0], int(p[1]), float(p[3])))
            except ValueError:
                pass
    return out


def classificar(serie):
    """Classifica a fase atual do ENSO a partir da série ONI."""
    if not serie:
        return None
    seas, ano, oni = serie[-1]
    # quantos trimestres seguidos (do fim para trás) estão do mesmo lado do limiar
    sinal = 1 if oni >= LIMIAR else -1 if oni <= -LIMIAR else 0
    seq = 0
    for _, _, v in reversed(serie):
        if sinal and (v >= LIMIAR if sinal > 0 else v <= -LIMIAR):
            seq += 1
        else:
            break
    fase = "El Niño" if sinal > 0 else "La Niña" if sinal < 0 else "Neutral"
    confirmado = sinal != 0 and seq >= MIN_SEQ
    tendencia = oni - serie[-4][2] if len(serie) >= 4 else 0.0  # variação em 3 trimestres
    return {
        "fase": fase,
        "oni": oni,
        "trimestre": f"{seas} {ano}",
        "intensidade": intensidade(oni),
        "confirmado": confirmado,
        "seq": seq,
        "tendencia": tendencia,
        "historico": serie[-24:],
        "impacto": IMPACTO_DF[fase],
    }


def rotulo(info):
    """Short label, e.g. 'strong El Niño' or 'weak La Niña (developing)'."""
    if info is None:
        return "no data"
    if info["fase"] == "Neutral":
        return "Neutral"
    return f"{info['intensidade']} {info['fase']}" + ("" if info["confirmado"] else " (developing)")


def obter():
    try:
        return classificar(baixar_oni())
    except Exception:
        return None
