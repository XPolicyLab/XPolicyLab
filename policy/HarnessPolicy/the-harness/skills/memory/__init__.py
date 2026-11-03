"""Read-only entries of the library's memory."""

import math


class Memory:
    def recall(self, query, k=5):
        """Library memory entries most relevant to `query` (word overlap), best first:
        [{"name", "text", "score"}]. Read-only; empty without a library."""
        import re

        def words(text):
            return set(re.findall(r"[a-z0-9]{2,}", str(text).lower()))

        wanted = words(query)
        scored = []
        for entry in self.memory:
            have = words(entry["name"].replace("_", " ") + " " + entry["text"])
            overlap = len(wanted & have)
            if overlap:
                scored.append({"name": entry["name"], "text": entry["text"],
                               "score": round(overlap / math.sqrt(len(have)), 4)})
        scored.sort(key=lambda item: -item["score"])
        return scored[:max(0, int(k))]
