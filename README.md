# Authentic RAG: BM25 + Vector + RRF

โปรเจกต์ Hands-on ตามสไลด์ Natural Language Processing สำหรับสร้าง Contextual RAG ด้วย OpenSearch โดยค้นหาแบบ Hybrid Search และรวมอันดับด้วย Reciprocal Rank Fusion (RRF)

## สถานะบนเครื่องนี้

- Python 3.12 virtual environment: `.venv/`
- Python dependencies: ติดตั้งแล้ว
- Docker Desktop: ติดตั้งและทำงานแล้ว
- OpenSearch 2.19.1: ทำงานที่ `http://localhost:9200`
- Embedding model `BAAI/bge-m3`: ดาวน์โหลดแล้ว (vector 1024 มิติ)
- Corpus ตัวอย่าง: อยู่ใน `corpus_input/`
- OpenRouter API key: ผู้ใช้ต้องตั้งค่าก่อนทำ Hands-on 1 และ 2

## โครงสร้างไฟล์

```text
.
├── corpus_input/                     # เอกสาร Markdown ตัวอย่าง 4 ไฟล์
├── authenticRAG.py                   # Hands-on 1: chunk, contextualize และ index
├── onlysearchAuthenticRAG.py         # Hands-on 2: BM25 + Vector + RRF + answer
├── docker-compose.yml                # OpenSearch 2.19.1
├── requirements.txt                  # Python dependencies
├── .env.example                      # ตัวอย่าง environment variables
└── authentic_rag_search_results.json # ผลลัพธ์ (สร้างหลังรัน Hands-on 2)
```

## 1. เปิด Python environment

```bash
source .venv/bin/activate
```

หากต้องสร้าง environment ใหม่:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. เปิด OpenSearch

เปิด Docker Desktop แล้วรัน:

```bash
docker compose up -d
curl http://localhost:9200
```

ตรวจ indices:

```bash
curl 'http://localhost:9200/_cat/indices?v'
```

หยุดระบบเมื่อเลิกใช้งาน:

```bash
docker compose stop
```

## 3. ตั้งค่า OpenRouter

สร้าง API key ที่ <https://openrouter.ai/keys> แล้วตั้งค่าใน Terminal เดียวกับที่จะรันโปรแกรม:

```bash
export OPENROUTER_API_KEY='ใส่คีย์ของคุณที่นี่'
export OPENROUTER_MODEL='qwen/qwen-2.5-72b-instruct'
```

อย่าใส่ API key จริงลงในไฟล์ที่ commit เข้า Git โดย `.env` ถูกเพิ่มไว้ใน `.gitignore` แล้ว

## 4. Hands-on 1: นำเอกสารเข้า OpenSearch

```bash
python authenticRAG.py
```

โปรแกรมจะ:

1. อ่านไฟล์ Markdown จาก `corpus_input/`
2. แบ่งเอกสารเป็น chunks
3. ใช้ OpenRouter สร้าง context ของแต่ละ chunk โดยอิงเอกสารเต็ม
4. สร้าง embedding ด้วย `BAAI/bge-m3`
5. บันทึกลง `anthropic-vector-index` และ `anthropic-bm25-index`
6. ทดลองค้นหาแบบ Hybrid Search

หมายเหตุ: corpus ตัวอย่างถูกแบ่งเป็น 53 chunks จึงเรียก OpenRouter อย่างน้อย 53 ครั้งเพื่อสร้าง context และอาจมีค่าใช้จ่าย API

## 5. Hands-on 2: ค้นหาและสร้างคำตอบ

```bash
python onlysearchAuthenticRAG.py
```

ผลลัพธ์จะถูกบันทึกเป็น `authentic_rag_search_results.json`

## 6. วัดผล Lab 1 และ Lab 2

หลังจาก OpenSearch ทำงานและมีข้อมูลในทั้งสอง index แล้ว ให้รัน:

```bash
python evaluate_performance.py
```

สคริปต์ประเมินดังนี้:

- **Lab 1:** จำนวนเอกสาร, ความครบถ้วนของ content/context, embedding dimension,
  ความตรงกันของ document IDs และขนาด index
- **Lab 2:** เปรียบเทียบ BM25, Vector และ RRF ด้วย Hit@5, Precision@5,
  Recall@5, MRR, nDCG@5 และ latency

ชุด relevance labels อยู่ใน `evaluation_qrels.json` และรายงานฉบับเต็มจะถูกบันทึกเป็น
`evaluation_report.json` หากเปลี่ยน corpus หรือวิธีแบ่ง chunk ต้องตรวจและแก้ qrels ใหม่

ผลการทดลองและข้อสรุปล่าสุดอ่านได้ที่ `EVALUATION.md`

## ตรวจสอบก่อนรัน

```bash
python -m py_compile authenticRAG.py onlysearchAuthenticRAG.py
curl 'http://localhost:9200/_cat/indices?v'
```

หลัง Hands-on 1 ค่า `docs.count` ของ `anthropic-vector-index` และ `anthropic-bm25-index` ควรมากกว่า 0

## เอกสารประกอบ

- `lab1-readme-from-text-to-vectordb.md`: คำอธิบายการนำเอกสารเข้า Vector Database
- `lab2-readme-from-vectordb-to-final-answer.md`: คำอธิบายการค้นหา BM25, Vector และ RRF
