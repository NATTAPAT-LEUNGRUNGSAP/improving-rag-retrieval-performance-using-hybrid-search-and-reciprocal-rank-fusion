import os

from onlysearchAuthenticRAG import AuthenticSearchRAG


THAI_BM25_INDEX = "anthropic-bm25-thai-index"


class ImprovedThaiSearchRAG(AuthenticSearchRAG):
    """Assignment extension: use Thai BM25 while keeping vector search, RRF and LLM unchanged."""

    def __init__(self, opensearch_host="localhost", opensearch_port=9200):
        super().__init__(opensearch_host, opensearch_port)
        self.bm25_index_name = THAI_BM25_INDEX


def main():
    if "OPENROUTER_API_KEY" not in os.environ:
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        print("Please set it with: export OPENROUTER_API_KEY='your_api_key_here'")
        return

    rag = ImprovedThaiSearchRAG()
    questions = [
        "โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?",
        "อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน",
        "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?",
        "ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?",
        "โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?",
    ]
    results = rag.search_multiple_questions(questions, k=5)
    rag.export_results_to_json(results, "improved_thai_rag_search_results.json")


if __name__ == "__main__":
    main()
