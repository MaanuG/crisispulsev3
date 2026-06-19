import time
from multi_source_engine import fetch_all_sources
from evidence_engine import enrich_relevance, enrich_with_stance
from state_store import TemporalEvidenceStore


class StreamEngine:

    def __init__(self):
        self.store = TemporalEvidenceStore()

    def process_batch(self, query: str):

        evidence = fetch_all_sources(query)

        if not evidence:
            return []

        evidence = enrich_relevance(query, evidence)
        evidence = enrich_with_stance(query, evidence)

        self.store.add_batch(evidence)

        return evidence

    def run_stream(self, query: str, interval: int = 30):

        """
        Continuously updates evidence every N seconds.
        """

        print("\n=== STREAM STARTED ===\n")

        while True:

            print(f"\n🔄 Updating evidence for: {query}")

            batch = self.process_batch(query)

            print(f"New evidence: {len(batch)} items")
            print(f"Total stored: {len(self.store.get_all())}")

            time.sleep(interval)