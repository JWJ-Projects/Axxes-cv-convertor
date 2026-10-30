"""Conservative taxonomy matching: related technologies are NOT synonyms."""
import csv
import re
from pathlib import Path

TAXONOMY_PATH = Path(__file__).resolve().parent / "data" / "taxonomy.csv"
# Deliberately reviewed aliases only. The CSV's consolidated-terms column also
# contains related products and must not be used blindly as an alias lookup.
SAFE_ALIASES = {"ts": "TypeScript", "golang": "Go", "js": "JavaScript"}
# The renderer contract has one category per skill. Resolve reviewed taxonomy
# duplicates explicitly instead of making CSV row order determine the result.
CATEGORY_OVERRIDES = {"mulesoft": "ERP/CRM & Integration"}


def key(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


class SkillNormalizer:
    def __init__(self, taxonomy_path: Path = TAXONOMY_PATH):
        self.entries: dict[str, tuple[str, str | None]] = {}
        with taxonomy_path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                name = row["Skill"].strip()
                if name:
                    normalized = key(name)
                    category = row["Category"].strip() or None
                    existing = self.entries.get(normalized)
                    if existing is None:
                        self.entries[normalized] = (name, category)
                    else:
                        existing_name, existing_category = existing
                        if existing_category and category and existing_category != category:
                            override = CATEGORY_OVERRIDES.get(normalized)
                            if override not in (existing_category, category):
                                raise ValueError(
                                    f"Conflicting taxonomy categories for {name!r}: "
                                    f"{existing_category!r} and {category!r}"
                                )
                            category = override
                        else:
                            category = existing_category or category
                        self.entries[normalized] = (existing_name, category)
        self.aliases = {
            alias: canonical for alias, canonical in SAFE_ALIASES.items()
            if key(canonical) in self.entries
        }

    def normalize(self, name: str) -> tuple[str, str | None]:
        normalized = key(name)
        normalized = key(self.aliases.get(normalized, normalized))
        return self.entries.get(normalized, (re.sub(r"\s+", " ", name.strip()), None))

    def mentions(self, text: str, name: str) -> bool:
        """Require token boundaries, including punctuation used in C++/C#/.NET."""
        variants = [name] + [a for a, n in self.aliases.items() if key(n) == key(name)]
        return any(re.search(r"(?<![\w+#.])" + re.escape(v) + r"(?![\w+#])", text, re.I)
                   for v in variants)

    def find_names(self, text: str) -> list[str]:
        # Short names such as Go, R and TS are ambiguous in ordinary prose.
        # Accept them when explicitly listed in a CV, not via automatic discovery.
        return [name for name, _ in self.entries.values()
                if len(name) > 2 and self.mentions(text, name)]
