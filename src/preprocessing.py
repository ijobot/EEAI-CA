"""Text preprocessing: de-duplication, template stripping, and noise removal.

Public entry points (all return a new DataFrame, never mutate in place):

    deduplicate(df) -> ticket-level de-duplication of interaction content.
        Training-time enhancement. No-op when there is no 'Ticket id' column
        (e.g. single new messages at inference), so it is safe to call in both
        pipelines.

    clean_text(df)  -> row-level lowercasing + boilerplate/noise removal, applied
        IDENTICALLY at training and inference. This is the part that guarantees
        training-serving consistency.

    preprocess(df)  -> convenience wrapper: deduplicate() then clean_text().

Run standalone to smoke-test:
    python -m src.preprocessing
"""

import logging
import re

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

# Raw-data column used only for training-time de-duplication.
TICKET_ID_COL = "Ticket id"


# --------------------------------------------------------------------------- #
# Pattern definitions
# --------------------------------------------------------------------------- #
# Multilingual customer-support signature/footer templates to strip.
_CU_TEMPLATES = [
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
    r"Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?",
    r"L'équipe d'assistance à la clientèle d'Aspiegel\,?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
    r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?",
    r"Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
]

# Email/thread split markers (used to segment interaction content).
_SPLIT_PATTERNS = [
    r"From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:",
    r"On.{30,60}wrote:",
    r"Re\s?:|RE\s?:",
    r"\*\*\*\*\*\(PERSON\) Support issue submit",
    r"\s?\*\*\*\*\*\(PHONE\)",
]

# Boilerplate phrases / dates / greetings removed from lowercased content.
# NOTE: the two implicit string-concatenation bugs from the prototype's noise_1
# list (missing commas) are fixed here — these are now four separate patterns.
_CONTENT_PHRASE_PATTERNS = [
    r"(?:from :)|(?:subject :)|(?:sent :)|(?:r\s*:)|(?:re\s*:)",
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)",
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    r"\d{2}(?::|.)\d{2}",
    r"(?:xxxxx@xxxx\.com)|(?:\*{5}\([a-z]+\))",
    r"dear (?:(?:customer)|(?:user))",
    r"dear",
    r"(?:hello)|(?:hallo)|(?:hi )|(?:hi there)",
    r"good morning",
    r"thank you for your patience (?:(?:during (?:our)? investigation)|(?:and cooperation))?",
    r"thank you for contacting us",
    r"thank you for your availability",
    r"thank you for providing us this information",
    r"thank you for contacting",
    r"thank you for reaching us (?:back)?",
    r"thank you for patience",
    r"thank you for (?:your)? reply",
    r"thank you for (?:your)? response",
    r"thank you for (?:your)? cooperation",
    r"thank you for providing us with more information",
    r"thank you very kindly",
    r"thank you(?: very much)?",
    r"i would like to follow up on the case you raised on the date",
    r"i will do my very best to assist you",          # was fused to the next line
    r"in order to give you the best solution",
    r"could you please clarify your request with following information:",  # was fused
    r"in this matter",
    r"we hope you(?:(?: are)|(?:'re)) doing (?:(?:fine)|(?:well))",
    r"i would like to follow up on the case you raised on",
    r"we apologize for the inconvenience",
    r"sent from my huawei (?:cell )?phone",
    r"original message",
    r"customer support team",
    r"(?:aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
    r"(?:aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
    r"canada, australia, new zealand and other countries",
]

# Ticket-summary-specific prefix noise (fwd/re markers, brackets, null/nan, etc.).
_SUMMARY_NOISE = (
    r"(?:sv\s*:)|(?:wg\s*:)|(?:ynt\s*:)|(?:fw(?:d)?\s*:)|(?:r\s*:)|(?:re\s*:)|"
    r"(?:\[|\])|(?:aspiegel support issue submit)|(?:null)|(?:nan)|"
    r"(?:(?:bonus place my )?support.pt 自动回复:)"
)


def _combine(patterns: list[str]) -> str:
    """Join patterns into a single non-capturing alternation.

    Reuses the combine-into-one-pattern technique the prototype already used for
    cu_pattern, but which it never applied to the ~30-item noise list. Compiling
    once (below) avoids recompiling these on every row.
    """
    return "|".join(f"(?:{p})" for p in patterns)


_CU_RE = re.compile(_combine(_CU_TEMPLATES))
_SPLIT_RE = re.compile(_combine(_SPLIT_PATTERNS))
_PHRASE_RE = re.compile(_combine(_CONTENT_PHRASE_PATTERNS))
_SUMMARY_RE = re.compile(_SUMMARY_NOISE)


# --------------------------------------------------------------------------- #
# De-duplication (training-time; ticket-aware)
# --------------------------------------------------------------------------- #
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated segments within each ticket's interaction content.

    Splits each interaction on email/thread markers, strips signature templates,
    and keeps only segments not already seen earlier in the same ticket. The
    logic is genuinely stateful (order-dependent within a ticket), so it stays a
    per-ticket loop rather than a vectorised op.

    Skipped entirely when there is no ticket id (inference), where a single
    message has nothing to de-duplicate against.
    """
    if TICKET_ID_COL not in df.columns:
        logger.info(
            "No '%s' column; skipping ticket-level de-duplication.", TICKET_ID_COL
        )
        return df

    df = df.copy()
    for ticket_id, group in df.groupby(TICKET_ID_COL):
        seen: set[str] = set()
        deduped: list[str] = []
        for content in group[Config.INTERACTION_CONTENT].astype(str):
            kept = []
            for segment in _SPLIT_RE.split(content):
                if segment is None:
                    continue
                segment = _CU_RE.sub("", segment.strip()).strip()
                if segment and segment not in seen:
                    seen.add(segment)
                    kept.append(segment)
            deduped.append(" ".join(kept))
        df.loc[group.index, Config.INTERACTION_CONTENT] = deduped

    return df


# --------------------------------------------------------------------------- #
# Row-level cleaning (training AND inference — must stay identical)
# --------------------------------------------------------------------------- #
def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip boilerplate/noise from both text columns.

    Applied the same way in training and inference to preserve training-serving
    consistency. Both columns now receive the same shared cleaning (the prototype
    cleaned summary and content with inconsistent rigor); the summary also gets
    its prefix-noise pass.
    """
    df = df.copy()
    for col in Config.TEXT_COLS:
        if col not in df.columns:
            continue

        s = df[col].fillna("").astype(str).str.lower()

        if col == Config.TICKET_SUMMARY:
            s = s.str.replace(_SUMMARY_RE, " ", regex=True)

        # Remove thread markers, signature templates, boilerplate phrases.
        s = s.str.replace(_SPLIT_RE, " ", regex=True)
        s = s.str.replace(_CU_RE, " ", regex=True)
        s = s.str.replace(_PHRASE_RE, " ", regex=True)

        # General normalisation: drop digits, non-alphanumerics, isolated chars.
        s = s.str.replace(r"\d+", " ", regex=True)
        s = s.str.replace(r"[^0-9a-z]+", " ", regex=True)
        s = s.str.replace(r"(?:\s|^).(?:\s|$)", " ", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()

        df[col] = s

    return df


# --------------------------------------------------------------------------- #
# Public wrapper
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing path: de-duplicate (if possible) then clean text."""
    df = deduplicate(df)
    df = clean_text(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from src.data_loader import load_training_data

    data = load_training_data()
    cleaned = preprocess(data)

    empty = (cleaned[Config.INTERACTION_CONTENT].str.len() == 0).sum()
    print(f"\nRows: {len(cleaned)} | empty interaction_content after cleaning: {empty}")
    for col in Config.TEXT_COLS:
        print(f"\n[{col}] sample after cleaning:")
        print(cleaned[col].head(3).to_string())