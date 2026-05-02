"""Central configuration — single source of truth for all tunable parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Resolve project root regardless of working directory ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Paths (derived from PROJECT_ROOT, not overridable via .env) ───────────
    @property
    def data_raw_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed"

    @property
    def db_path(self) -> Path:
        return PROJECT_ROOT / "db"

    @property
    def sqlite_path(self) -> Path:
        return PROJECT_ROOT / "db" / "metadata.db"

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512

    @field_validator("llm_temperature")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("llm_temperature must be in [0.0, 2.0]")
        return v

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "nomic-embed-text"
    embedding_fallback: str = "all-MiniLM-L6-v2"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 400       # tokens per chunk
    chunk_overlap: int = 80     # overlap tokens between consecutive chunks
    min_chunk_size: int = 50    # discard chunks shorter than this

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 400)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 5
    max_context_tokens: int = 2000

    # ── Similarity thresholds ─────────────────────────────────────────────────
    similarity_threshold_low: float = 0.30   # below → IDK (no LLM call)
    similarity_threshold_mid: float = 0.45   # below → low-confidence prefix

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_collection_name: str = "wikipedia_rag"

    # ── Entity type constants (do not change — used as metadata filter values) ─
    entity_type_person: Literal["person"] = "person"
    entity_type_place: Literal["place"] = "place"


# Singleton — import `settings` everywhere, never instantiate Settings directly
settings = Settings()


# ── Entity catalog ────────────────────────────────────────────────────────────
# `wikipedia_title` is the exact Wikipedia page title used by the API.
# It may differ from `entity_name` (e.g. display name vs. page slug).

PEOPLE_LIST: list[dict] = [
    {"entity_id": "albert_einstein",        "entity_name": "Albert Einstein",         "wikipedia_title": "Albert Einstein",         "entity_type": "person"},
    {"entity_id": "marie_curie",            "entity_name": "Marie Curie",             "wikipedia_title": "Marie Curie",             "entity_type": "person"},
    {"entity_id": "leonardo_da_vinci",      "entity_name": "Leonardo da Vinci",       "wikipedia_title": "Leonardo da Vinci",       "entity_type": "person"},
    {"entity_id": "william_shakespeare",    "entity_name": "William Shakespeare",     "wikipedia_title": "William Shakespeare",     "entity_type": "person"},
    {"entity_id": "ada_lovelace",           "entity_name": "Ada Lovelace",            "wikipedia_title": "Ada Lovelace",            "entity_type": "person"},
    {"entity_id": "nikola_tesla",           "entity_name": "Nikola Tesla",            "wikipedia_title": "Nikola Tesla",            "entity_type": "person"},
    {"entity_id": "lionel_messi",           "entity_name": "Lionel Messi",            "wikipedia_title": "Lionel Messi",            "entity_type": "person"},
    {"entity_id": "cristiano_ronaldo",      "entity_name": "Cristiano Ronaldo",       "wikipedia_title": "Cristiano Ronaldo",       "entity_type": "person"},
    {"entity_id": "taylor_swift",           "entity_name": "Taylor Swift",            "wikipedia_title": "Taylor Swift",            "entity_type": "person"},
    {"entity_id": "frida_kahlo",            "entity_name": "Frida Kahlo",             "wikipedia_title": "Frida Kahlo",             "entity_type": "person"},
    {"entity_id": "stephen_hawking",        "entity_name": "Stephen Hawking",         "wikipedia_title": "Stephen Hawking",         "entity_type": "person"},
    {"entity_id": "napoleon_bonaparte",     "entity_name": "Napoleon Bonaparte",      "wikipedia_title": "Napoleon",               "entity_type": "person"},
    {"entity_id": "mahatma_gandhi",         "entity_name": "Mahatma Gandhi",          "wikipedia_title": "Mahatma Gandhi",          "entity_type": "person"},
    {"entity_id": "nelson_mandela",         "entity_name": "Nelson Mandela",          "wikipedia_title": "Nelson Mandela",          "entity_type": "person"},
    {"entity_id": "cleopatra",              "entity_name": "Cleopatra",               "wikipedia_title": "Cleopatra",               "entity_type": "person"},
    {"entity_id": "isaac_newton",           "entity_name": "Isaac Newton",            "wikipedia_title": "Isaac Newton",            "entity_type": "person"},
    {"entity_id": "wolfgang_amadeus_mozart","entity_name": "Wolfgang Amadeus Mozart", "wikipedia_title": "Wolfgang Amadeus Mozart", "entity_type": "person"},
    {"entity_id": "martin_luther_king_jr",  "entity_name": "Martin Luther King Jr.",  "wikipedia_title": "Martin Luther King Jr.", "entity_type": "person"},
    {"entity_id": "pablo_picasso",          "entity_name": "Pablo Picasso",           "wikipedia_title": "Pablo Picasso",           "entity_type": "person"},
    {"entity_id": "aristotle",              "entity_name": "Aristotle",               "wikipedia_title": "Aristotle",               "entity_type": "person"},
]

PLACES_LIST: list[dict] = [
    {"entity_id": "eiffel_tower",               "entity_name": "Eiffel Tower",              "wikipedia_title": "Eiffel Tower",              "entity_type": "place"},
    {"entity_id": "great_wall_of_china",        "entity_name": "Great Wall of China",       "wikipedia_title": "Great Wall of China",       "entity_type": "place"},
    {"entity_id": "taj_mahal",                  "entity_name": "Taj Mahal",                 "wikipedia_title": "Taj Mahal",                 "entity_type": "place"},
    {"entity_id": "grand_canyon",               "entity_name": "Grand Canyon",              "wikipedia_title": "Grand Canyon",              "entity_type": "place"},
    {"entity_id": "machu_picchu",               "entity_name": "Machu Picchu",              "wikipedia_title": "Machu Picchu",              "entity_type": "place"},
    {"entity_id": "colosseum",                  "entity_name": "Colosseum",                 "wikipedia_title": "Colosseum",                 "entity_type": "place"},
    {"entity_id": "hagia_sophia",               "entity_name": "Hagia Sophia",              "wikipedia_title": "Hagia Sophia",              "entity_type": "place"},
    {"entity_id": "statue_of_liberty",          "entity_name": "Statue of Liberty",         "wikipedia_title": "Statue of Liberty",         "entity_type": "place"},
    {"entity_id": "pyramids_of_giza",           "entity_name": "Pyramids of Giza",          "wikipedia_title": "Giza pyramids",             "entity_type": "place"},
    {"entity_id": "mount_everest",              "entity_name": "Mount Everest",             "wikipedia_title": "Mount Everest",             "entity_type": "place"},
    {"entity_id": "stonehenge",                 "entity_name": "Stonehenge",                "wikipedia_title": "Stonehenge",                "entity_type": "place"},
    {"entity_id": "angkor_wat",                 "entity_name": "Angkor Wat",                "wikipedia_title": "Angkor Wat",                "entity_type": "place"},
    {"entity_id": "petra",                      "entity_name": "Petra",                     "wikipedia_title": "Petra, Jordan",             "entity_type": "place"},
    {"entity_id": "amazon_rainforest",          "entity_name": "Amazon Rainforest",         "wikipedia_title": "Amazon rainforest",         "entity_type": "place"},
    {"entity_id": "yellowstone_national_park",  "entity_name": "Yellowstone National Park", "wikipedia_title": "Yellowstone National Park", "entity_type": "place"},
    {"entity_id": "venice",                     "entity_name": "Venice",                    "wikipedia_title": "Venice",                    "entity_type": "place"},
    {"entity_id": "acropolis_of_athens",        "entity_name": "Acropolis of Athens",       "wikipedia_title": "Acropolis of Athens",       "entity_type": "place"},
    {"entity_id": "tokyo",                      "entity_name": "Tokyo",                     "wikipedia_title": "Tokyo",                     "entity_type": "place"},
    {"entity_id": "istanbul",                   "entity_name": "Istanbul",                  "wikipedia_title": "Istanbul",                  "entity_type": "place"},
    {"entity_id": "pompeii",                    "entity_name": "Pompeii",                   "wikipedia_title": "Pompeii",                   "entity_type": "place"},
]

ENTITY_CATALOG: list[dict] = PEOPLE_LIST + PLACES_LIST

# Lookup tables (built once at import time, used by router.py)
ENTITY_ID_TO_META: dict[str, dict] = {e["entity_id"]: e for e in ENTITY_CATALOG}
ENTITY_NAME_LOWER: dict[str, dict] = {e["entity_name"].lower(): e for e in ENTITY_CATALOG}
