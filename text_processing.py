"""text_processing.py — Text normalisation and LSE word-order rewriting.

LSE word order differs fundamentally from Spanish:
  - Topic-Comment, not Subject-Verb-Object
  - Question words move to the END of the utterance
  - Articles are dropped (handled by matcher.IGNORABLE)
  - Inflected verb forms → infinitives (doler/duele/dueles → doler)

The pipeline for LSE:
  clean_text(text, use_spanish=True)
    → strip accents, lowercase, remove punctuation
    → normalise variant spellings
    → normalise inflected verbs to infinitives
  reorder_lse(words)
    → move question words to end (topic-comment structure)
    → result fed to matcher.match_phrases()
"""

import string
import unicodedata
from typing import List

# ── English normalisations ────────────────────────────────────────────────────
ENGLISH_NORMALIZATIONS = {
    # ── Greetings ─────────────────────────────────────────────────────────────
    "thanks": "thank you", "thank": "thank you", "thankyou": "thank you",
    "ty": "thank you", "thx": "thank you", "hey": "hello", "hi": "hello",
    "howdy": "hello", "hiya": "hello", "whats": "what is",
    "what's": "what is", "i'm": "my name is", "im": "my name is",

    # ── Body parts — spoken variants → dictionary key ─────────────────────────
    "ears":         "ear",
    "eardrum":      "ear",
    "hearing":      "ear",
    "nostrils":     "nose",
    "nasal":        "nose",
    "sinus":        "nose",
    "sinuses":      "nose",
    "tonsils":      "throat",
    "tonsil":       "throat",
    "larynx":       "throat",
    "pharynx":      "throat",
    "lips":         "mouth",
    "lip":          "mouth",
    "jaw":          "mouth",
    "gums":         "teeth",
    "tooth":        "teeth",
    "molar":        "teeth",
    "molars":       "teeth",
    "cervical":     "neck",
    "forehead":     "head",
    "skull":        "head",
    "temple":       "head",
    "temples":      "head",

    # ── Symptoms — spoken variants → dictionary key ───────────────────────────
    "hurt":         "pain",
    "hurts":        "pain",
    "hurting":      "pain",
    "ache":         "pain",
    "aches":        "pain",
    "aching":       "pain",
    "sore":         "pain",
    "painful":      "pain",
    "discomfort":   "pain",
    "coughing":     "cough",
    "coughs":       "cough",
    "coughed":      "cough",
    "temperature":  "fever",
    "hot":          "fever",
    "feverish":     "fever",
    "dizziness":    "dizzy",
    "dizzy spell":  "dizzy",
    "lightheaded":  "dizzy",
    "light-headed": "dizzy",
    "vertigo":      "dizzy",
    "ringing":      "tinnitus",
    "buzzing":      "tinnitus",
    "ringing in ears": "tinnitus",
    "bleeding":     "blood",
    "bleed":        "blood",
    "bleeds":       "blood",
    "bloody":       "blood",
    "infected":     "infection",
    "inflamed":     "swelling",
    "inflammation": "swelling",
    "swollen":      "swelling",
    "swelled":      "swelling",
    "swells":       "swelling",
    "mucous":       "mucus",
    "phlegm":       "mucus",
    "congestion":   "mucus",
    "congested":    "mucus",
    "runny":        "mucus",
    "snot":         "mucus",
    "hoarse":       "voice",
    "hoarseness":   "voice",
    "lost my voice": "voice",
    "can't speak":  "voice",
    "voiceless":    "voice",

    # ── Clinical actions — spoken variants ────────────────────────────────────
    "breathing":    "breathe",
    "breath":       "breathe",
    "inhale":       "breathe",
    "exhale":       "breathe",
    "swallowing":   "swallow",
    "swallowed":    "swallow",
    "hard to swallow": "swallow",
    "looking":      "look",
    "turning":      "turn",
    "rotate":       "turn",

    # ── Daily verbs — spoken variants ─────────────────────────────────────────
    "eating":       "eat",
    "ate":          "eat",
    "eats":         "eat",
    "food":         "eat",
    "drinking":     "drink",
    "drank":        "drink",
    "drinks":       "drink",
    "water":        "drink",
    "sleeping":     "sleep",
    "slept":        "sleep",
    "sleeps":       "sleep",
    "tired":        "sleep",
    "walking":      "walk",
    "walked":       "walk",
    "walks":        "walk",
    "sitting":      "sit",
    "sat":          "sit",
    "sits":         "sit",
    "seated":       "sit",
}

# ── Spanish normalisations ────────────────────────────────────────────────────
# Maps spoken/inflected Spanish → the dictionary key that will match a clip.
# Two kinds:
#   A) Common mis-recognitions / variant spellings
#   B) Inflected verb forms → infinitive (LSE uses citation forms only)
SPANISH_NORMALIZATIONS = {
    # ── Variant spellings / speech recogniser noise ───────────────────────────
    "oido":         "oido",      # accent already stripped by _strip_accents,
    "infeccion":    "infeccion", # but listed here in case recogniser drops accents
    "hinchazon":    "hinchazon", # before we do — no-op but explicit

    # ── Verb inflections → infinitive ─────────────────────────────────────────
    # respirar
    "respira":      "respirar",
    "respire":      "respirar",
    "respiro":      "respirar",
    "respirad":     "respirar",
    # tragar
    "traga":        "tragar",
    "trague":       "tragar",
    "trago":        "tragar",
    # abrir
    "abre":         "abrir",
    "abra":         "abrir",
    "abro":         "abrir",
    # mirar
    "mira":         "mirar",
    "mire":         "mirar",
    "miro":         "mirar",
    # girar
    "gira":         "girar",
    "gire":         "girar",
    "giro":         "girar",
    # comer
    "come":         "comer",
    "comes":        "comer",
    "como":         "comer",    # "como" as verb ("I eat") → comer clip
    "coma":         "comer",
    # beber
    "bebe":         "beber",
    "bebes":        "beber",
    "bebo":         "beber",
    "beba":         "beber",
    # dormir
    "duerme":       "dormir",
    "duermes":      "dormir",
    "duermo":       "dormir",
    "duerma":       "dormir",
    # caminar
    "camina":       "caminar",
    "caminas":      "caminar",
    "camino":       "caminar",
    # sentar(se)
    "sienta":       "sentar",
    "sientas":      "sentar",
    "siento":       "sentar",    # "I sit" not "I feel" in this context
    "sientate":     "sentar",
    "sentarse":     "sentar",
    "sientese":     "sentar",    # formal imperative
    # doler (maps to "dolor" clip — the noun is what's in the dictionary)
    "duele":        "dolor",
    "duelen":       "dolor",
    "dueles":       "dolor",
    "doler":        "dolor",
    "dolor de":     "dolor",
    # tos variants
    "tose":         "tos",
    "toser":        "tos",
    # fiebre
    "fiebre alta":  "fiebre",
    # mareo
    "mareado":      "mareo",
    "mareada":      "mareo",
    # hinchazon
    "hinchado":     "hinchazon",
    "hinchada":     "hinchazon",
    "inflamado":    "hinchazon",
    "inflamacion":  "hinchazon",
    # sangre
    "sangrar":      "sangre",
    "sangra":       "sangre",
    # voz
    "ronco":        "voz",
    "ronca":        "voz",
    "afonía":       "voz",
    "afonia":       "voz",
    # moco
    "mocos":        "moco",
    "mucosidad":    "moco",
    "secrecion":    "moco",
}


def _strip_accents(text: str) -> str:
    """Remove diacritics: 'gargánta' → 'garganta', 'oído' → 'oido'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _remove_consecutive_duplicates(words: List[str]) -> List[str]:
    return [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]


def clean_text(text: str, use_spanish: bool = False) -> List[str]:
    """
    Full normalisation pipeline.
    Returns a word list ready for matcher.match_phrases().

    Steps:
      1. Strip accent marks  (oído → oido, infección → infeccion)
      2. Lowercase + remove punctuation
      3. Apply multi-word normalisations first (longest match)
      4. Apply single-word normalisations  (duele → dolor, respira → respirar)
      5. Remove consecutive duplicates
    """
    norms = SPANISH_NORMALIZATIONS if use_spanish else ENGLISH_NORMALIZATIONS

    text = _strip_accents(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Multi-word phrases (e.g. "dolor de" → "dolor", "fiebre alta" → "fiebre")
    for src, dst in sorted(norms.items(), key=lambda x: -len(x[0])):
        if " " in src and src in text:
            text = text.replace(src, dst)

    words = []
    for word in text.split():
        words.extend(norms[word].split() if word in norms else [word])

    return _remove_consecutive_duplicates(words)
