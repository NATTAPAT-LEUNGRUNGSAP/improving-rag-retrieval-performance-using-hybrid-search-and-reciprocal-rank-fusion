# รายงานการวัดผล Authentic RAG

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

ชุด qrels นี้กำหนดด้วยการตรวจเนื้อหาใน corpus โดยมนุษย์ หากเปลี่ยน corpus หรือวิธีแบ่ง
chunk ต้องตรวจ relevance labels ใหม่ และคะแนนชุดนี้ยังไม่ใช่การประเมินคำตอบเชิงข้อเท็จจริง
แบบผู้เชี่ยวชาญทางการแพทย์

## วิธีรันวัดผลซ้ำ

```bash
source .venv/bin/activate
docker compose up -d
python evaluate_performance.py
```

