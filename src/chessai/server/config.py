"""Plafonds de sécurité du serveur. Ils vivent ici, côté serveur, et ne sont pas négociables."""

# Plafond de calcul par analyse : le client peut demander moins, jamais plus.
MAX_NODES = 800
MAX_TIME_S = 2.0

# Nombre d'analyses exécutées en même temps. Au-delà, on attend au plus MAX_TIME_S puis 503.
MAX_CONCURRENT = 2

# Limitation de débit par adresse IP (syntaxe slowapi / limits).
RATE_LIMIT = "30/minute"

# Taille maximale d'un corps de requête, en octets. Un FEN tient en moins de 100.
MAX_BODY_BYTES = 4096
