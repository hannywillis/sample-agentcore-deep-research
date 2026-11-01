"""Evidence-grounded, context-aware agricultural decision support.

Implements the three-layer architecture of the CNPq PROAFRICA proposal
"Evidence-Grounded and Context-Aware AI for Agricultural Decision Support in
African and Brazilian Smallholder Farming Systems":

* WP1 Knowledge layer  -> :mod:`agri_advisor.rag`, :mod:`agri_advisor.nlp`,
  :mod:`agri_advisor.validation`, :mod:`agri_advisor.governance`
* WP2 Predictive layer -> :mod:`agri_advisor.predict`,
  :mod:`agri_advisor.engine`
* WP3 Impact layer     -> :mod:`agri_advisor.interface` (tiered access,
  usage/feedback logging for impact evaluation)
"""

__version__ = "0.1.0"
