# รายงานการวัดผล Improving RAG Retrieval Performance Using Hybrid Search and Reciprocal Rank Fusion

วัดผลเมื่อวันที่ 17 ตุลาคม 2026 จาก corpus 53 chunks บน OpenSearch 2.19.1
ผลฉบับเต็มสำหรับตรวจสอบซ้ำอยู่ใน `evaluation_report.json`

## Lab 1: คุณภาพการนำข้อมูลเข้า Index

| รายการ | ผลลัพธ์ |
|---|---:|
| Vector documents | 53 |
| BM25 documents | 53 |
| Document IDs ตรงกันทั้งสอง index | ผ่าน |
| เอกสารมี content ครบ | 100% |
| เอกสารมี contextualized content ครบ | 100% |
| Embedding dimension | 1,024 |
| Vector index size | 1,584,248 bytes |
| BM25 index size | 256,951 bytes |
| สถานะรวม | **ผ่าน** |

ค่าเฉลี่ยต่อ chunk:

- เนื้อหา 1,633.60 ตัวอักษร
- Context 224.32 ตัวอักษร

## Lab 2: คุณภาพการค้นคืนข้อมูล

ประเมิน 5 คำถามด้วย relevance labels ใน `evaluation_qrels.json` และคืนผล Top-5
โดยแต่ละ retriever ดึง candidate 10 รายการก่อนรวมอันดับด้วย RRF (`rrf_k=60`)

| วิธี | Hit@5 | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Vector | 1.000 | 0.360 | 0.933 | 1.000 | 0.862 |
| RRF | 1.000 | 0.360 | 0.933 | 1.000 | 0.862 |

### Ranking ที่ได้จริง

ตารางนี้แสดง Top-5 document IDs ของแต่ละวิธีจากการทดลองจริง เครื่องหมาย `—`
หมายถึง retriever ไม่คืนผลลัพธ์

| คำถาม | BM25 Top-5 | Vector Top-5 | RRF Top-5 |
|---|---|---|---|
| Q1 ความแตกต่างของหัดและหัดเยอรมัน | — | doc_1, doc_2, doc_3, doc_8, doc_6 | doc_1, doc_2, doc_3, doc_8, doc_6 |
| Q2 สาเหตุและการป้องกัน | — | doc_2, doc_0, doc_1, doc_13, doc_15 | doc_2, doc_0, doc_1, doc_13, doc_15 |
| Q3 อันตรายต่อหญิงตั้งครรภ์ | — | doc_6, doc_1, doc_2, doc_4, doc_11 | doc_6, doc_1, doc_2, doc_4, doc_11 |
| Q4 โอกาสติดเชื้อหลังฉีดวัคซีน | — | doc_13, doc_2, doc_11, doc_1, doc_6 | doc_13, doc_2, doc_11, doc_1, doc_6 |
| Q5 ผลกระทบต่อสาธารณสุขและเศรษฐกิจ | — | doc_1, doc_2, doc_6, doc_3, doc_10 | doc_1, doc_2, doc_6, doc_3, doc_10 |

ตัวอย่างคะแนน RRF ของ Q1:

| Document | BM25 Rank | Vector Rank | RRF Rank | RRF Score |
|---|---:|---:|---:|---:|
| doc_1 | — | 1 | 1 | 1/60 = 0.016667 |
| doc_2 | — | 2 | 2 | 1/61 = 0.016393 |
| doc_3 | — | 3 | 3 | 1/62 = 0.016129 |
| doc_8 | — | 4 | 4 | 1/63 = 0.015873 |
| doc_6 | — | 5 | 5 | 1/64 = 0.015625 |

เนื่องจาก BM25 ไม่มี ranking ให้รวม คะแนน RRF จึงมาจาก Vector เพียงรายการเดียว
และลำดับไม่เปลี่ยน หาก document เดียวกันได้อันดับแรกจากทั้ง BM25 และ Vector คะแนนจะ
เป็น `1/60 + 1/60 = 0.033333`

### เวลาเฉลี่ย

วัดซ้ำคำถามละ 3 รอบหลัง warm-up โมเดลแล้ว เวลาจริงอาจเปลี่ยนตามเครื่อง

| ขั้นตอน | Mean | P95 |
|---|---:|---:|
| BM25 search | 3.855 ms | 6.992 ms |
| Query embedding | 42.430 ms | 57.053 ms |
| Vector search | 14.581 ms | 19.720 ms |
| RRF fusion | 0.010 ms | 0.013 ms |
| Hybrid end-to-end | 60.878 ms | 82.647 ms |

เวลาโหลดโมเดลครั้งแรก 9,327.038 ms ไม่รวมอยู่ใน latency ต่อคำถาม

## การสร้างคำตอบ

- มีผลลัพธ์ครบ 5 คำถาม
- ทุกคำถามมีคำตอบที่ไม่ว่าง
- ทุกคำถามมีเอกสาร Top-5 ครบ
- ไม่พบ API error ในไฟล์ผลลัพธ์

## ข้อสรุปและข้อจำกัด

Vector retrieval พบเอกสารที่เกี่ยวข้องใน Top-5 ครบทุกคำถาม และมี Recall@5 เฉลี่ย
0.933 ส่วน BM25 ไม่คืนเอกสารสำหรับคำถามภาษาไทยชุดนี้ จึงทำให้ RRF ได้อันดับเดียวกับ
Vector แทนที่จะเกิดประโยชน์จากการรวมสอง retrievers

สาเหตุมีแนวโน้มมาจาก BM25 index ใช้ Standard Analyzer ซึ่งไม่เหมาะกับข้อความภาษาไทย
ที่ไม่มีการเว้นวรรคระหว่างคำ งานต่อไปควรทดลอง Thai tokenizer หรือ character n-gram analyzer
แล้วสร้าง BM25 index ใหม่ ก่อนวัดผลเปรียบเทียบอีกครั้ง

การทดลองส่วนปรับปรุงได้ดำเนินการแล้วโดยสร้าง `anthropic-bm25-thai-index`
แยกจาก Baseline ผลเต็มอยู่ใน `improvement_evaluation_report.json` และคำอธิบาย
เปรียบเทียบตาม Assignment อยู่ใน `ASSIGNMENT_COMPARISON.md`

ชุด qrels นี้กำหนดด้วยการตรวจเนื้อหาใน corpus โดยมนุษย์ หากเปลี่ยน corpus หรือวิธีแบ่ง
chunk ต้องตรวจ relevance labels ใหม่ และคะแนนชุดนี้ยังไม่ใช่การประเมินคำตอบเชิงข้อเท็จจริง
แบบผู้เชี่ยวชาญทางการแพทย์

## วิธีรันวัดผลซ้ำ

```bash
source .venv/bin/activate
docker compose up -d
python evaluate_performance.py
```
