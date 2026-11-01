"""Fixed interface phrases per language (used by SMS/USSD/voice templates).

These are short, safety-critical strings that must not depend on an LLM.
TODO(Task 1.6 / D1.6): have each string reviewed by native speakers at the
pilot sites. Zarma-Songhay ("dje") currently falls back to French, the
official language in Niger, until validated translations are supplied.
"""

from __future__ import annotations

PHRASES: dict[str, dict[str, str]] = {
    "en": {
        "confidence": "Confidence",
        "high": "high",
        "medium": "medium",
        "low": "low",
        "consult_expert": "Please confirm with an extension officer before acting.",
        "sources": "Sources",
        "prediction": "Prediction",
        "no_evidence": (
            "I could not find validated guidance for this question. "
            "Please contact your extension officer."
        ),
        "translation_unavailable": "(Showing English: translation service offline.)",
        "ussd_welcome": "Farm advice: 1.Ask a question 2.Crop risk 3.Talk to expert",
        "ussd_ask": "Type your question:",
        "ussd_crop": "Crop? 1.Maize 2.Wheat 3.Sorghum 4.Rice 5.Millet",
        "ussd_expert": "An extension officer will call you back. Thank you.",
        "more_sms": "Reply MORE for details.",
    },
    "sw": {
        "confidence": "Uhakika",
        "high": "juu",
        "medium": "wastani",
        "low": "chini",
        "consult_expert": "Tafadhali thibitisha na afisa ugani kabla ya kuchukua hatua.",
        "sources": "Vyanzo",
        "prediction": "Utabiri",
        "no_evidence": (
            "Sikupata ushauri uliothibitishwa kwa swali hili. "
            "Tafadhali wasiliana na afisa ugani."
        ),
        "translation_unavailable": "(Kiingereza: huduma ya tafsiri haipatikani.)",
        "ussd_welcome": "Ushauri wa kilimo: 1.Uliza swali 2.Hatari ya zao 3.Mtaalamu",
        "ussd_ask": "Andika swali lako:",
        "ussd_crop": "Zao? 1.Mahindi 2.Ngano 3.Mtama 4.Mpunga 5.Uwele",
        "ussd_expert": "Afisa ugani atakupigia simu. Asante.",
        "more_sms": "Jibu MORE kwa maelezo zaidi.",
    },
    "fr": {
        "confidence": "Confiance",
        "high": "élevée",
        "medium": "moyenne",
        "low": "faible",
        "consult_expert": "Veuillez confirmer avec un agent de vulgarisation avant d'agir.",
        "sources": "Sources",
        "prediction": "Prévision",
        "no_evidence": (
            "Je n'ai pas trouvé de conseil validé pour cette question. "
            "Veuillez contacter votre agent de vulgarisation."
        ),
        "translation_unavailable": "(Anglais : service de traduction indisponible.)",
        "ussd_welcome": "Conseil agricole: 1.Poser une question 2.Risque 3.Expert",
        "ussd_ask": "Écrivez votre question :",
        "ussd_crop": "Culture? 1.Maïs 2.Blé 3.Sorgho 4.Riz 5.Mil",
        "ussd_expert": "Un agent de vulgarisation vous rappellera. Merci.",
        "more_sms": "Répondez MORE pour plus de détails.",
    },
    "ha": {
        "confidence": "Tabbaci",
        "high": "babba",
        "medium": "matsakaici",
        "low": "kadan",
        "consult_expert": (
            "Don Allah ka tabbatar da jami'in fadakarwa kafin ka dauki mataki."
        ),
        "sources": "Majiyoyi",
        "prediction": "Hasashe",
        "no_evidence": (
            "Ban sami shawarar da aka tabbatar ba game da wannan tambaya. "
            "Don Allah ka tuntubi jami'in fadakarwa."
        ),
        "translation_unavailable": "(Turanci: sabis na fassara ba ya aiki.)",
        "ussd_welcome": "Shawarar noma: 1.Yi tambaya 2.Hadarin amfanin gona 3.Masani",
        "ussd_ask": "Rubuta tambayarka:",
        "ussd_crop": "Amfanin gona? 1.Masara 2.Alkama 3.Dawa 4.Shinkafa 5.Gero",
        "ussd_expert": "Jami'in fadakarwa zai kira ka. Na gode.",
        "more_sms": "Amsa MORE don karin bayani.",
    },
    "pt": {
        "confidence": "Confiança",
        "high": "alta",
        "medium": "média",
        "low": "baixa",
        "consult_expert": "Por favor, confirme com um técnico de extensão antes de agir.",
        "sources": "Fontes",
        "prediction": "Previsão",
        "no_evidence": (
            "Não encontrei orientação validada para esta pergunta. "
            "Por favor, contate o seu técnico de extensão."
        ),
        "translation_unavailable": "(Em inglês: serviço de tradução indisponível.)",
        "ussd_welcome": "Assistência agrícola: 1.Perguntar 2.Risco da lavoura 3.Técnico",
        "ussd_ask": "Escreva sua pergunta:",
        "ussd_crop": "Cultura? 1.Milho 2.Trigo 3.Sorgo 4.Arroz 5.Milheto",
        "ussd_expert": "Um técnico de extensão vai ligar para você. Obrigado.",
        "more_sms": "Responda MORE para detalhes.",
    },
}

_FALLBACK = {"dje": "fr"}


def phrase(key: str, language: str) -> str:
    lang = language if language in PHRASES else _FALLBACK.get(language, "en")
    return PHRASES[lang].get(key, PHRASES["en"][key])
