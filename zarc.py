"""ZARC (Zoneamento Agrícola de Risco Climático - MAPA) para Brasília/DF.

Usa o recorte data/zarc_brasilia.csv (gerado por scripts/atualizar_zarc.py).
Cada linha da tábua de risco diz, para cultura x ciclo x solo, o risco de perda em cada
decêndio de plantio (dec1 = 1-10/jan ... dec36 = 21-31/dez): 20, 30 ou 40 (%); 0 = não indicado.
"""
import functools
import pathlib
import pandas as pd

ARQ = pathlib.Path(__file__).resolve().parent / "data" / "zarc_brasilia.csv"
DECS = [f"dec{i}" for i in range(1, 37)]
MESES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CICLO = {"20": "Group I (early)", "21": "Group II (medium)", "22": "Group III (late)", "24": "Group IV"}

# Solo: o produtor escolhe pela "cara" do solo; o ZARC usa classes de água disponível (AD1 a AD6).
# Se a safra usar a classificação antiga por textura (1, 2, 3), cai nela como alternativa.
SOLOS = {
    "Don't know (use medium soil)": ["13", "14", "2"],
    "Sandy, dries out fast (AD1)": ["11", "1"],
    "Sandy with some clay (AD2)": ["12", "1"],
    "Medium texture (AD3)": ["13", "2"],
    "Medium to clayey (AD4)": ["14", "2"],
    "Clayey, holds water well (AD5)": ["15", "3"],
    "Very clayey, red latosol (AD6)": ["16", "3"],
}

# Duração aproximada da soja do plantio à colheita, em decêndios, por grupo do ZARC.
# Aproximação para checar se dá tempo do milho safrinha; ajuste conforme a cultivar.
SOJA_DECS_ATE_COLHER = {"20": 10, "21": 12, "22": 13}  # ~100, ~120, ~130 dias


def carregar():
    if not ARQ.exists():
        return None
    d = pd.read_csv(ARQ, sep=";", dtype=str)
    d[DECS] = d[DECS].fillna("0").astype(int)
    return d


def safras(d):
    return sorted(d["SafraIni"].unique(), reverse=True)


def nome_dec(n):
    """dec1 -> 'Jan 1–10', dec3 -> 'Jan 21–end'."""
    n = (n - 1) % 36 + 1
    mes, parte = MESES[(n - 1) // 3], (n - 1) % 3
    return f"{mes} {['1–10', '11–20', '21–end'][parte]}"


def decendio_de(data):
    return (data.month - 1) * 3 + min((data.day - 1) // 10, 2) + 1


def _solo_rows(d, cultura, solo, safra, manejo="Sequeiro"):
    q = d[(d["Nome_cultura"] == cultura) & (d["SafraIni"] == safra) & (d["Nome_Outros_Manejos"] == manejo)]
    for cod in SOLOS[solo]:
        r = q[q["Cod_Solo"] == cod]
        if len(r):
            return r
    return q.iloc[0:0]


def riscos(d, cultura, solo, safra, manejo="Sequeiro"):
    """{cod_ciclo: [risco dec1..dec36]} para a cultura no solo escolhido (menor risco se houver repetição)."""
    out = {}
    for ciclo, g in _solo_rows(d, cultura, solo, safra, manejo).groupby("Cod_Ciclo"):
        v = g[DECS].replace(0, 999).min().replace(999, 0)
        out[ciclo] = v.tolist()
    return out


def janelas(lista, max_risco=40):
    """Junta decêndios seguidos (passando de dez para jan) em janelas: [(ini, fim, pior_risco)]."""
    ordem = list(range(19, 37)) + list(range(1, 19))  # começa em julho p/ não quebrar out->jan
    out, cur = [], None
    for n in ordem:
        r = lista[n - 1]
        if 0 < r <= max_risco:
            if cur and cur[1] == n - 1 or (cur and cur[1] == 36 and n == 1):
                cur = (cur[0], n, max(cur[2], r))
            else:
                if cur: out.append(cur)
                cur = (n, n, r)
        else:
            if cur: out.append(cur); cur = None
    if cur: out.append(cur)
    return out


def texto_janelas(lista, max_risco=40):
    js = janelas(lista, max_risco)
    if not js:
        return "no recommended window"
    return "; ".join(f"{nome_dec(a)} to {nome_dec(b)} (risk up to {r}%)" for a, b, r in js)


def tolerancia(enso_info):
    """Risco máximo aceitável pelo ZARC conforme a fase do ENSO (heurística do app)."""
    if enso_info and enso_info["fase"] == "El Niño" and enso_info["intensidade"] in ("moderate", "strong", "very strong"):
        return 20
    return 30


def combinar_soja_milho(d, solo, safra, segunda="Milho 2ª Safra", max_risco=30):
    """Procura combinações soja -> 2ª safra que caibam no ZARC dentro do risco aceito.

    Para cada grupo de soja e decêndio de plantio permitido, estima a colheita e verifica se o
    decêndio seguinte está liberado para a 2ª safra. Devolve lista ordenada pelo menor risco somado.
    """
    soja, milho = riscos(d, "Soja", solo, safra), riscos(d, segunda, solo, safra)
    combos = []
    for gs, rs in soja.items():
        dur = SOJA_DECS_ATE_COLHER.get(gs)
        if not dur:
            continue
        for n in range(1, 37):
            if not (0 < rs[n - 1] <= max_risco):
                continue
            m = (n - 1 + dur) % 36 + 1  # decêndio da colheita = plantio do milho
            for gm, rm in milho.items():
                if 0 < rm[m - 1] <= max_risco:
                    combos.append(dict(soja_ciclo=CICLO.get(gs, gs), soja_dec=n, soja_risco=rs[n - 1],
                                       milho_ciclo=CICLO.get(gm, gm), milho_dec=m, milho_risco=rm[m - 1]))
    return sorted(combos, key=lambda c: (c["soja_risco"] + c["milho_risco"], c["soja_dec"] < 19, c["soja_dec"]))


def calendario(d, solo, safra, culturas):
    """Tabela cultura x decêndio (jul a jun) com o MENOR risco entre os grupos de ciclo."""
    ordem = list(range(19, 37)) + list(range(1, 19))
    linhas = {}
    for c in culturas:
        rs = riscos(d, c, solo, safra)
        if not rs:
            continue
        melhor = [min([r[n - 1] for r in rs.values() if r[n - 1] > 0], default=0) for n in ordem]
        linhas[c] = melhor
    cols = [nome_dec(n) for n in ordem]
    return pd.DataFrame(linhas, index=cols).T if linhas else pd.DataFrame()


# ---------------- Consulta por mês, para o catálogo de culturas do app ----------------
# Cultura do catálogo do app -> nomes na tábua do ZARC. Primeira e segunda safra contam: vale a de menor risco no mês.
DO_CATALOGO = {
    "Soybean": ("Soja",),
    "Corn": ("Milho 1ª Safra", "Milho 2ª Safra"),
    "Beans": ("Feijão", "Feijão 2ª Safra"),
    "Sorghum": ("Sorgo Granífero", "Sorgo Granífero 2ª Safra"),
}

_BASE = []


def base():
    """Tábua de Brasília lida uma vez por processo (None se o arquivo não existir)."""
    if not _BASE:
        _BASE.append(carregar())
    return _BASE[0]


@functools.lru_cache(maxsize=4096)
def risco_mes(cultura, solo, mes, irrigado=False):
    """Menor risco do ZARC de Brasília para plantar `cultura` (nome do catálogo do app) no mês `mes` (1-12).

    None: a cultura não tem ZARC (ou não há base). Senão (risco, decêndio): risco 0 = zoneada, mas não indicada
    nesse mês. Com irrigação usa as linhas de "Irrigado" quando a cultura tem; senão as de sequeiro.
    """
    nomes, d = DO_CATALOGO.get(cultura), base()
    if not nomes or d is None:
        return None
    safra, zoneada, melhor = safras(d)[0], False, None
    for nome in nomes:
        for manejo in (("Irrigado", "Sequeiro") if irrigado else ("Sequeiro",)):
            rs = riscos(d, nome, solo, safra, manejo)
            if not rs:
                continue
            zoneada = True
            for lista in rs.values():
                for n in range(mes * 3 - 2, mes * 3 + 1):
                    r = lista[n - 1]
                    if r and (melhor is None or r < melhor[0]):
                        melhor = (r, n)
            break  # o primeiro manejo com dados vale (irrigado, quando existe)
    if not zoneada:
        return None
    return melhor or (0, None)
