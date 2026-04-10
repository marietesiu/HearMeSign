"""matcher.py — Greedy longest-match clip lookup with LSE word-order support.

For LSE (Spanish mode):
  1. text_processing.clean_text() strips accents, normalises verbs
  2. reorder_lse() moves question words to end (LSE topic-comment order)
  3. match_phrases() looks up each word/phrase against LSE_DICT
  4. IGNORABLE words (articles, fillers) are silently skipped

LSE articles (el/la/los/las/un/una) are in IGNORABLE because they are
NOT signed in LSE — signing them would be wrong, not just redundant.
"""

import os
from typing import Dict, List
from config import ASL_CLIPS_FOLDER

# ── Words to silently skip ────────────────────────────────────────────────────
# English fillers
_IGNORABLE_EN = {
    # Articles and determiners
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their",
    # Pronouns
    "i", "me", "we", "you", "he", "she", "it", "they", "them",
    # Common verbs that add no sign meaning
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "feel", "feels", "feeling", "felt",
    "get", "gets", "getting", "got",
    "think", "seems", "seem",
    # Modal verbs
    "can", "cant", "could", "would", "should", "will", "might", "may",
    # Prepositions
    "in", "on", "at", "to", "from", "with", "about", "of", "for",
    "by", "into", "onto", "over", "under", "through", "around",
    # Fillers and qualifiers
    "um", "uh", "like", "just", "really", "very", "so", "quite",
    "now", "today", "please", "here", "there", "up", "down",
    "doing", "going", "also", "too", "bit", "little", "some",
    "kind", "sort", "type", "lot", "lots", "much", "many",
    # Connector words
    "and", "but", "or", "when", "if", "because", "since",
    "while", "then", "also", "again",
    # Medical filler phrases people naturally say
    "experiencing", "having", "suffering", "complaining",
    "notice", "noticed", "suddenly", "lately", "recently",
    "always", "sometimes", "often", "constantly", "keep",
    "keeps", "bad", "badly", "severe", "mild", "worse", "better",
    "little", "slight", "slight", "chronic", "acute",
}

# Spanish — articles and fillers that are NOT signed in LSE
_IGNORABLE_ES = {
    # Articles — NOT signed in LSE (this is a grammar rule, not a missing clip)
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Common fillers / connectors that carry no sign equivalent
    "hoy", "ahora", "muy", "bien", "pues", "este", "eh",
    "favor",        # "por favor" → "por" is signed, "favor" alone is not
    "se",           # reflexive marker — incorporated into verb sign
    "le", "les",    # indirect object clitics — not signed separately
    "nos",          # "we" object clitic — drop when context is clear
    "esto", "eso", "aquello",   # demonstratives rarely signed in isolation
    "mas",          # "más" as filler; signed when meaning "more" explicitly
    "tambien",      # "también" — can be dropped in rapid LSE
    "ya",           # filler "ya" (affirmative particle) — not signed
    "si",           # "sí" (yes) — keep only if intentional; filtered here
                    # because speech recogniser often inserts it as filler
    "no",           # negation IS signed in LSE — but after the verb.
                    # Keeping it in ignorable for now because word-order
                    # placement of NO requires sentence-level analysis we
                    # don't yet do. TODO: implement negation placement.
}

IGNORABLE = _IGNORABLE_EN | _IGNORABLE_ES


def match_phrases(words: List[str], active_dict: Dict[str, str],
                  use_spanish: bool = False) -> List[str]:
    """
    Greedy longest-match lookup against active_dict.
    Returns ordered list of .mp4 file paths.

    For Spanish (LSE) mode, applies reorder_lse() before matching so that
    question words appear at the end of the clip sequence.
    """
    if not words or not active_dict:
        return []

    max_len = max(len(k.split()) for k in active_dict)
    videos  = []
    i       = 0

    # Determine clip folder: LSE uses lse_clips/, ASL uses asl_clips/
    if use_spanish:
        clip_folder = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "lse_clips")
    else:
        clip_folder = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), ASL_CLIPS_FOLDER)

    while i < len(words):
        matched = False

        # Try longest phrase first, shrink until match or single word
        for size in range(min(max_len, len(words) - i), 0, -1):
            phrase = " ".join(words[i: i + size])
            if phrase in active_dict:
                path = os.path.join(clip_folder, active_dict[phrase])
                videos.append(path)
                print(f"[matcher] '{phrase}' → {active_dict[phrase]}")
                i      += size
                matched = True
                break

        if not matched:
            word = words[i]
            if word in IGNORABLE:
                print(f"[matcher] Ignored (LSE grammar): '{word}'")
            else:
                # Fallback: fingerspell letter by letter
                found = False
                for letter in word:
                    p = os.path.join(clip_folder, f"{letter}.mp4")
                    if os.path.exists(p):
                        videos.append(p)
                        found = True
                    else:
                        print(f"[matcher] No clip for '{letter}'")
                if not found:
                    print(f"[matcher] Skipped (no clip, no fingerspell): '{word}'")
            i += 1

    return videos
