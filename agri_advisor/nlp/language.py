"""Lightweight language identification and agricultural query expansion.

General-purpose detectors (e.g. ``langdetect``) do not support Hausa or
Zarma-Songhay and are unreliable on the short, code-mixed messages typical of
SMS. We therefore score each message against curated function-word and
agricultural-term lists, then fall back to ``langdetect`` (if installed) for
ambiguous English/French/Portuguese/Swahili cases.

TODO(Task 1.6): extend and validate the word lists with native speakers from
each pilot site; add Zarma-Songhay terms from the CRESA team.
"""

from __future__ import annotations

import re
import unicodedata

from agri_advisor.config import SUPPORTED_LANGUAGES

_TOKEN = re.compile(r"[\w'’]+", re.UNICODE)

_MARKERS: dict[str, set[str]] = {
    "en": {
        "the",
        "and",
        "how",
        "what",
        "when",
        "which",
        "should",
        "my",
        "is",
        "are",
        "can",
        "do",
        "i",
        "to",
        "of",
        "for",
        "with",
        "plant",
        "crop",
        "farm",
    },
    "sw": {
        "na",
        "ya",
        "wa",
        "kwa",
        "ni",
        "je",
        "vipi",
        "nini",
        "gani",
        "lini",
        "nifanye",
        "shamba",
        "mahindi",
        "mbolea",
        "kupanda",
        "mvua",
        "wadudu",
        "yangu",
        "wangu",
        "kuna",
        "hii",
        "kwenye",
        "mazao",
        "mkulima",
        "nipande",
        "ninawezaje",
        "udongo",
        "mbegu",
        "viwavijeshi",
        "ukame",
        "mtama",
        "ngano",
    },
    "fr": {
        "le",
        "la",
        "les",
        "des",
        "du",
        "est",
        "comment",
        "quand",
        "quel",
        "quelle",
        "mon",
        "ma",
        "mes",
        "je",
        "pour",
        "que",
        "dans",
        "avec",
        "maïs",
        "semer",
        "engrais",
        "pluie",
        "sécheresse",
        "champ",
        "culture",
        "faire",
    },
    "ha": {
        "da",
        "ne",
        "ce",
        "ta",
        "yaya",
        "menene",
        "masara",
        "gona",
        "zan",
        "ina",
        "kuma",
        "shuka",
        "taki",
        "ruwa",
        "wane",
        "yaushe",
        "amfanin",
        "gonar",
        "na",
        "ake",
        "kwari",
        "gero",
        "dawa",
        "shinkafa",
        "alkama",
        "manomi",
    },
    "pt": {
        "o",
        "os",
        "as",
        "um",
        "uma",
        "como",
        "quando",
        "qual",
        "meu",
        "minha",
        "não",
        "para",
        "é",
        "plantar",
        "milho",
        "adubo",
        "chuva",
        "seca",
        "solo",
        "lavoura",
        "praga",
        "fazer",
        "devo",
        "posso",
        "no",
        "na",
        "do",
        "da",
    },
    "dje": {  # Zarma-Songhay - experimental, to be curated with CRESA.
        "ay",
        "ni",
        "ga",
        "kaŋ",
        "hayni",
        "fari",
        "hari",
        "hansi",
        "foo",
        "boro",
        "manti",
        "wala",
        "koyne",
    },
}

# Weighted so that distinctive words beat shared short words ("na", "da").
_DISTINCTIVE: dict[str, set[str]] = {
    "sw": {"mahindi", "nifanye", "shamba", "mbolea", "viwavijeshi", "kupanda"},
    "ha": {"masara", "yaya", "menene", "gona", "gonar", "taki", "kwari", "gero"},
    "fr": {"maïs", "comment", "quand", "engrais", "sécheresse", "semer"},
    "pt": {"milho", "como", "quando", "adubo", "devo", "não", "lavoura"},
    "dje": {"hayni", "kaŋ"},
    "en": {"the", "how", "what", "should", "maize"},
}

# Local-language agricultural terms -> English, used to expand retrieval
# queries when no LLM is available to translate the query.
AGRI_GLOSSARY: dict[str, dict[str, str]] = {
    "sw": {
        "mahindi": "maize",
        "ngano": "wheat",
        "mtama": "sorghum",
        "mpunga": "rice",
        "mchele": "rice",
        "uwele": "millet",
        "viwavijeshi": "fall armyworm",
        "wadudu": "pest insects",
        "mbolea": "fertilizer",
        "kupanda": "planting",
        "nipande": "plant planting",
        "mvua": "rain",
        "ukame": "drought",
        "udongo": "soil",
        "magonjwa": "disease",
        "kuhifadhi": "storage",
        "sumukuvu": "aflatoxin",
        "mbegu": "seed variety",
        "palizi": "weeding",
        "kutu": "rust",
        "chokaa": "lime liming",
        "kudhibiti": "control manage",
        "kuzuia": "prevent control",
        "kukausha": "drying",
        "nafaka": "grain",
    },
    "ha": {
        "masara": "maize",
        "alkama": "wheat",
        "dawa": "sorghum",
        "shinkafa": "rice",
        "gero": "millet",
        "taki": "fertilizer",
        "kwari": "pest insects",
        "tsutsa": "larvae armyworm",
        "shuka": "planting",
        "ruwan": "rain",
        "fari": "drought",
        "kasa": "soil",
        "cuta": "disease",
        "iri": "seed variety",
        "ajiya": "storage",
        "kuduji": "striga",
        "magance": "control manage",
        "busar": "drying",
        "hatsi": "grain",
    },
    "fr": {
        "maïs": "maize",
        "blé": "wheat",
        "sorgho": "sorghum",
        "riz": "rice",
        "mil": "millet",
        "engrais": "fertilizer",
        "semis": "planting",
        "semer": "planting",
        "pluie": "rain",
        "sécheresse": "drought",
        "sol": "soil",
        "ravageur": "pest",
        "ravageurs": "pest",
        "chenille": "armyworm larvae",
        "légionnaire": "fall armyworm",
        "maladie": "disease",
        "rouille": "rust",
        "stockage": "storage",
        "semences": "seed",
        "chaulage": "liming",
        "striga": "striga",
        "stocker": "storage store",
        "aflatoxines": "aflatoxin",
        "aflatoxine": "aflatoxin",
        "lutter": "control",
        "contrôler": "control",
        "sécher": "drying",
        "séchage": "drying",
    },
    "pt": {
        "milho": "maize",
        "trigo": "wheat",
        "sorgo": "sorghum",
        "arroz": "rice",
        "milheto": "millet",
        "adubo": "fertilizer",
        "fertilizante": "fertilizer",
        "plantio": "planting",
        "plantar": "planting",
        "chuva": "rain",
        "seca": "drought",
        "solo": "soil",
        "praga": "pest",
        "lagarta": "armyworm larvae",
        "doença": "disease",
        "ferrugem": "rust",
        "armazenamento": "storage",
        "sementes": "seed",
        "calagem": "liming",
        "brusone": "rice blast",
        "controlar": "control",
        "combater": "control",
        "aflatoxina": "aflatoxin",
        "secagem": "drying",
        "armazenar": "storage store",
    },
    "dje": {"hayni": "millet", "fari": "field farm"},
}

_CROP_TERMS: dict[str, str] = {
    "maize": "maize",
    "corn": "maize",
    "wheat": "wheat",
    "sorghum": "sorghum",
    "rice": "rice",
    "millet": "millet",
}
for _lang_terms in AGRI_GLOSSARY.values():
    for _term, _en in _lang_terms.items():
        if _en in _CROP_TERMS:
            _CROP_TERMS[_term] = _en


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(unicodedata.normalize("NFC", text))]


def detect_language(text: str, default: str = "en") -> str:
    """Return an ISO 639 code from :data:`SUPPORTED_LANGUAGES`."""
    toks = _tokens(text)
    if not toks:
        return default
    scores = dict.fromkeys(_MARKERS, 0.0)
    for tok in toks:
        for lang, words in _MARKERS.items():
            if tok in words:
                scores[lang] += 1.0
            if tok in _DISTINCTIVE.get(lang, set()):
                scores[lang] += 2.0
    best = max(scores, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] == 0 or (ranked[0] - ranked[1] < 1.0 and len(toks) > 3):
        try:  # pragma: no cover - optional dependency
            from langdetect import DetectorFactory, detect

            DetectorFactory.seed = 0
            guess = detect(text)
            if guess in SUPPORTED_LANGUAGES:
                return guess
        except Exception:  # noqa: S110 - best-effort fallback
            pass
        if ranked[0] == 0:
            return default
    return best


def tokenize(text: str) -> list[str]:
    return _tokens(text)


def expand_query(text: str, language: str, english_only: bool = False) -> str:
    """Map known local agricultural terms to English for retrieval.

    With ``english_only`` the result contains only the English glossary terms
    (when at least one was found), which retrieves better against English
    documents than the diluted mixed-language string.
    """
    glossary = AGRI_GLOSSARY.get(language, {})
    extra = [glossary[t] for t in _tokens(text) if t in glossary]
    if not extra:
        return text
    return " ".join(extra) if english_only else f"{text} {' '.join(extra)}"


def detect_crop(text: str) -> str | None:
    for tok in _tokens(text):
        if tok in _CROP_TERMS:
            return _CROP_TERMS[tok]
    return None
