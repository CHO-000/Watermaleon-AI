# Watermelon Leaf Disease API

โมเดล Deep Learning (EfficientNet-B0) สำหรับจำแนกภาพใบแตงโม 4 คลาส พร้อม FastAPI สำหรับเชื่อมหน้าเว็บ

## คลาสที่ทำนาย

| ID | ความหมาย |
|---|---|
| `Anthracnose` | โรคแอนแทรคโนส |
| `Downy_Mildew` | โรคราน้ำค้าง |
| `Healthy` | ใบปกติ |
| `Mosaic_Virus` | โรคไวรัสใบด่าง |

## เปิดใน VS Code และรัน API

เปิดโฟลเดอร์ `watermelon-disease-api` ใน VS Code แล้วกด Run and Debug > **FastAPI: Watermelon Disease** หรือรันคำสั่งด้านล่าง เครื่องนี้ใช้ Python 3.14 และ PyTorch ที่รองรับ CUDA และมีแพ็กเกจ API ใน `.vendor` สำหรับรันได้ทันทีบนเครื่องนี้

```powershell
py run_api.py
```

ทดสอบที่ `http://127.0.0.1:8000/docs` หรือ `http://127.0.0.1:8000/health` หากไฟล์ `model.pt` อยู่ในโฟลเดอร์นี้ API จะโหลดโมเดลเมื่อเริ่มทำงาน

เมื่อนำโปรเจกต์ไปเครื่องอื่น ให้สร้าง virtual environment และติดตั้ง `py -m pip install -r requirements.txt` ก่อน `.vendor` ถูกละจาก Git เพื่อไม่ผูกกับ Python/Windows รุ่นของเครื่องนี้

ตัวอย่างเรียกจาก frontend:

```javascript
const form = new FormData();
form.append("file", imageFile);
const response = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  body: form,
});
if (!response.ok) throw new Error(await response.text());
const result = await response.json();
console.log(result.class_name_th, result.confidence, result.scores);
```

API รับไฟล์ JPEG, PNG, WebP ไม่เกิน 10 MB และตอบ JSON ที่มี `class_id`, `class_name_th`, `confidence`, `scores`, `note` ค่า `confidence` คือคะแนน softmax ของโมเดล ยังไม่ได้ผ่านการปรับเทียบเป็นความน่าจะเป็นทางคลินิกหรือเกษตรกรรม

กำหนด origin ของหน้าเว็บได้ผ่านตัวแปร `CORS_ORIGINS` คั่นหลายค่าโดย comma ค่าเริ่มต้นคือ `http://localhost:3000,http://localhost:5173` (รวมถึง `127.0.0.1` ต้องระบุเองถ้าหน้าเว็บใช้ชื่อนั้น) กำหนดตำแหน่งโมเดลผ่าน `MODEL_PATH` ได้

## ฝึกใหม่

```powershell
py train.py --archive "C:\path\to\archive.zip" --architecture efficientnet_b0 --strong-augment --epochs 12 --batch-size 16
```

สคริปต์ใช้เฉพาะโฟลเดอร์ `Original Image/Watermelon` ใน ZIP และแบ่งข้อมูลตาม **ลำดับหมายเลขภาพภายในแต่ละคลาส**: train 70%, validation 15%, test 15% เพื่อลดภาพที่ถ่ายต่อเนื่องหลุดข้ามชุด ภาพ augmented ที่ให้มาถูกตัดออก เพราะเป็นสำเนาดัดแปลงจากภาพต้นฉบับและเสี่ยงข้อมูลรั่วข้ามชุด ส่วน augmentation สำหรับการฝึกสร้างสดจากภาพ train เท่านั้น

โมเดลใช้ pretrained EfficientNet-B0 แล้วฝึกปรับน้ำหนักทั้งหมด พร้อมสุ่มครอป แสง สี และมุมเฉพาะภาพ train เลือก checkpoint ด้วย macro F1 บน validation และประเมิน test หลังฝึกเสร็จ ผลทดสอบแยกตามคลาสอยู่ใน `metrics.json` ข้อมูลทดสอบมาจากแหล่งเดียวกับชุดฝึก จึงยังต้องทดสอบภาพจากสวนจริง/โทรศัพท์และสภาพแสงหลากหลายก่อนใช้ตัดสินใจเรื่องโรคพืชจริง ภาพพืชชนิดอื่นหรือโรคที่ไม่อยู่ใน 4 คลาสอาจยังถูกบังคับให้ทายเป็นคลาสใดคลาสหนึ่ง

## ผลทดสอบที่ได้บนเครื่องนี้

- ภาพต้นฉบับ 1,155 ภาพ: train 809, validation 173, test 173
- Test accuracy **89.0%**, macro F1 **89.8%**
- โรคราน้ำค้าง: recall **82.5%** (พบถูก 47 จาก 57 ภาพ) ยังไม่เหมาะให้ใช้ผลทำนายเป็นคำวินิจฉัยสุดท้าย
- โมเดลรุ่นแรก MobileNetV3 Small ทำได้ 82.1% บน test ชุดเดียวกัน รุ่น EfficientNet-B0 เลือกจากคะแนน validation ที่สูงกว่า แล้วทดสอบบน test
- ทดสอบ HTTP จริงแล้ว: `/health`, `/classes`, `/docs`, `/predict` ตอบ 200; ไฟล์ภาพเสียตอบ 422; ชนิดไฟล์ไม่รองรับตอบ 415; CORS สำหรับ `localhost:5173` ผ่าน

## ไฟล์สำคัญ

- `model.pt` น้ำหนักโมเดลพร้อมใช้งาน
- `metrics.json` ผล validation/test และ confusion matrix
- `train.py` เตรียมข้อมูลและฝึกโมเดล
- `app.py` FastAPI
- `model.py` โครงสร้างโมเดลและ preprocessing ร่วมกัน
