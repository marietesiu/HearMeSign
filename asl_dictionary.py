"""asl_dictionary.py — Sign clip dictionaries for ASL and LSE.

LSE GRAMMAR NOTES (critical — LSE ≠ Spanish word order)
═══════════════════════════════════════════════════════════

1. TOPIC-COMMENT structure, not Subject-Verb-Object
   Spanish: "Me duele la garganta"
   LSE:      GARGANTA  DOLOR  YO

2. QUESTION WORDS go at the END, not the beginning
   Spanish: "¿Dónde te duele?"
   LSE:      DOLOR  DÓNDE
   text_processing.py reorders question words to the end automatically.

3. NO ARTICLES — el/la/los/las/un/una are NOT signed in LSE
   They are listed in matcher.IGNORABLE and silently dropped.

4. NEGATION — NO comes AFTER the verb (not coded here, handled by text flow)

5. TIME MARKERS come FIRST (before topic and comment)

6. PRONOUNS are dropped when clear from context; kept only for emphasis

7. INFLECTED verb forms ("duele", "tengo", "tienes" etc.) are normalised
   to infinitives in text_processing.SPANISH_NORMALIZATIONS before lookup,
   because LSE uses base/citation forms of verbs without inflection.
"""

# ── ASL (English) — mirrors LSE_DICT signs in English ────────────────────────
# Keys   = normalised spoken English (lowercase).
# Values = mp4 filename inside asl_clips/ folder.
ASL_DICT = {
    # ── ENT body parts ────────────────────────────────────────────────────────
    "ear":          "ear.mp4",
    "nose":         "nose.mp4",
    "throat":       "throat.mp4",
    "mouth":        "mouth.mp4",
    "tongue":       "tongue.mp4",
    "teeth":        "teeth.mp4",
    "head":         "head.mp4",
    "neck":         "neck.mp4",

    # ── ENT symptoms ──────────────────────────────────────────────────────────
    "pain":         "pain.mp4",
    "cough":        "cough.mp4",
    "fever":        "fever.mp4",
    "dizzy":        "dizzy.mp4",
    "tinnitus":     "tinnitus.mp4",
    "blood":        "blood.mp4",
    "infection":    "infection.mp4",
    "swelling":     "swelling.mp4",
    "mucus":        "mucus.mp4",
    "voice":        "voice.mp4",

    # ── Clinical actions ──────────────────────────────────────────────────────
    "breathe":      "breathe.mp4",
    "swallow":      "swallow.mp4",
    "look":         "look.mp4",
    "turn":         "turn.mp4",

    # ── Basic daily verbs ─────────────────────────────────────────────────────
    "eat":          "eat.mp4",
    "drink":        "drink.mp4",
    "sleep":        "sleep.mp4",
    "walk":         "walk.mp4",
    "sit":          "sit.mp4",
}

# ── LSE (Spanish) — ENT clinical focus ───────────────────────────────────────
# Keys   = normalised spoken Spanish (accents stripped, lowercase).
# Values = mp4 filename inside lse_clips/ folder.
# 34 signs total: body parts + symptoms + clinical actions + basic verbs.
LSE_DICT = {
    # ── ENT body parts ────────────────────────────────────────────────────────
    "oreja":    "oreja.mp4",
    "oido":     "oido.mp4",
    "nariz":    "nariz.mp4",
    "garganta": "garganta.mp4",
    "boca":     "boca.mp4",
    "lengua":   "lengua.mp4",
    "dientes":  "dientes.mp4",
    "cabeza":   "cabeza.mp4",
    "cuello":   "cuello.mp4",

    # ── ENT symptoms ──────────────────────────────────────────────────────────
    "dolor":        "dolor.mp4",
    "tos":          "tos.mp4",
    "fiebre":       "fiebre.mp4",
    "mareo":        "mareo.mp4",
    "zumbido":      "zumbido.mp4",
    "sangre":       "sangre.mp4",
    "infeccion":    "infeccion.mp4",
    "hinchazon":    "hinchazon.mp4",
    "moco":         "moco.mp4",
    "voz":          "voz.mp4",

    # ── Clinical actions ──────────────────────────────────────────────────────
    "respirar": "respirar.mp4",
    "tragar":   "tragar.mp4",
    "mirar":    "mirar.mp4",
    "girar":    "girar.mp4",

    # ── Basic daily verbs ─────────────────────────────────────────────────────
    "comer":    "comer.mp4",
    "beber":    "beber.mp4",
    "dormir":   "dormir.mp4",
    "caminar":  "caminar.mp4",
    "sentar":   "sentar.mp4",
}