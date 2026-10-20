# การเปรียบเทียบ Baseline และระบบปรับปรุง

เอกสารนี้ตอบข้อ (3) ของ Assignment โดยเปรียบเทียบระบบตามสไลด์กับระบบที่ปรับปรุง
Thai BM25 และ Weighted Reciprocal Rank Fusion (Weighted RRF)

## (1) Baseline ตามสไลด์หน้า 22 และ 25

- แบ่งเอกสารและสร้าง contextualized chunks
- สร้าง embedding ด้วย `BAAI/bge-m3`
- ใช้ BM25 index ชื่อ `anthropic-bm25-index` ซึ่งกำหนด Standard analyzer
- ใช้ Vector Search จาก `anthropic-vector-index`
- รวมอันดับด้วย RRF โดยให้น้ำหนัก BM25 และ Vector เท่ากัน
- ใช้ LLM เดิมสร้างคำตอบสุดท้าย

โค้ด Baseline คือ `authenticRAG.py` และ `onlysearchAuthenticRAG.py`

## (2) ระบบปรับปรุง

ระบบปรับปรุงเปลี่ยนเฉพาะส่วน Search และ Scoring:

1. สร้าง `anthropic-bm25-thai-index` ด้วย Thai analyzer โดยไม่แก้หรือลบ Baseline
2. นำเอกสาร 53 chunks ชุดเดิมมา index ใหม่ จึงไม่มีค่าใช้จ่ายสร้าง context ซ้ำ
3. ใช้ Thai BM25 ร่วมกับ Vector Search เดิม
4. ทดลองทั้ง RRF น้ำหนักเท่ากันและ Weighted RRF ซึ่งกำหนดน้ำหนัก Thai BM25 = 0.25
   และ Vector = 1.0
5. ไม่เปลี่ยน LLM ที่สร้างคำตอบสุดท้าย

โค้ดส่วนปรับปรุงคือ `build_thai_bm25_index.py`, `onlysearchImprovedRAG.py`
และ `evaluate_improvement.py`

## (3) ต่างกันตรงไหนและดีขึ้นอย่างไร

ตัวอย่างข้อความ `อาการของโรคหัดเยอรมัน`:

| ระบบ | Analyzer | Token ที่ได้ |
|---|---|---|
| Baseline | Standard | `อาการของโรคหัดเยอรมัน` |
| ปรับปรุง | Thai | `อาการ`, `โรค`, `หัดเยอรมัน` |

Standard analyzer ไม่แยกคำภาษาไทยที่เขียนติดกัน ทำให้คำใน query ไม่ตรงกับคำในเอกสาร
และ BM25 ไม่คืนผลลัพธ์ในชุดทดลองนี้ ส่วน Thai analyzer แยกคำภาษาไทยได้ จึงทำให้
lexical retrieval กลับมาทำงาน

ประเมินด้วย corpus 53 chunks, คำถาม 5 ข้อ, Top-5, candidate 10 รายการต่อ retriever
และ relevance labels ชุดเดียวกัน:

| วิธี | Hit@5 | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Baseline BM25 (Standard) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Thai BM25 | 1.000 | 0.240 | 0.667 | 0.557 | 0.534 |
| Vector | 1.000 | 0.360 | 0.933 | 1.000 | 0.862 |
| Baseline RRF | 1.000 | 0.360 | 0.933 | 1.000 | 0.862 |
| Thai BM25 + Vector + RRF น้ำหนักเท่ากัน | 1.000 | 0.280 | 0.733 | 0.900 | 0.729 |
| Thai BM25 + Vector + Weighted RRF | **1.000** | **0.400** | **1.000** | 0.767 | 0.799 |

ผลที่ดีขึ้นชัดเจนคือ:

- Thai BM25 เพิ่ม Recall@5 จาก 0.000 เป็น 0.667
- Weighted RRF เพิ่ม Precision@5 จาก 0.360 เป็น 0.400
- Weighted RRF เพิ่ม Recall@5 จาก 0.933 เป็น 1.000
- เวลาเฉลี่ย end-to-end เพิ่มจาก 58.686 ms เป็น 61.023 ms หรือประมาณ 2.337 ms

ข้อจำกัดคือ RRF แบบน้ำหนักเท่ากันทำให้ผลรวมแย่กว่า Vector เพียงอย่างเดียว และแม้
Weighted RRF เพิ่ม Precision/Recall แต่ MRR และ nDCG@5 ยังต่ำกว่า Baseline แสดงว่า
พบเอกสาร relevant ครบขึ้นแต่ตำแหน่งของเอกสาร relevant แรกและลำดับโดยรวมยังไม่ดีขึ้น
ทุก metric จึงควรรายงานผลทั้งด้านที่ดีขึ้นและ trade-off นี้ตามจริง

น้ำหนัก 0.25/1.0 เป็นค่าทดลองเริ่มต้นกับชุดคำถามขนาดเล็ก ไม่ควรอ้างว่าเป็นค่าที่ดีที่สุด
สำหรับข้อมูลทั่วไป งานต่อไปควรแยก validation queries ออกจาก test queries ก่อนปรับน้ำหนัก

## วิธีทดลองซ้ำ

```bash
source .venv/bin/activate
docker compose up -d
python build_thai_bm25_index.py --recreate
python evaluate_improvement.py
```

รายงานแบบ machine-readable อยู่ใน `improvement_evaluation_report.json`
