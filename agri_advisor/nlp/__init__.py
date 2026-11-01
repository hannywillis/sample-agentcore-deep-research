"""WP1 - NLP knowledge translation and multilingual support."""

from agri_advisor.nlp.language import detect_crop, detect_language, expand_query
from agri_advisor.nlp.translator import KnowledgeTranslator, TranslationResult

__all__ = [
    "KnowledgeTranslator",
    "TranslationResult",
    "detect_crop",
    "detect_language",
    "expand_query",
]
