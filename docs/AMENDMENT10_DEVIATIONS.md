# Amendment 10: deviations from the registered materials

1. 24 October 2026, Part B. scripts/pythia_familiarity.py stopped on some Pythia tokenizers that define no padding,
   end-of-text or unknown token ("Asking to pad but the tokenizer does not have a padding token"). The fallback now uses
   token id 0 as the padding token. Padded positions are masked and only the last real token is read, so scores are
   unaffected (batched and single-item scores were checked to agree exactly). The deprecated argument torch_dtype was
   renamed dtype. No score had been produced by the affected models.
