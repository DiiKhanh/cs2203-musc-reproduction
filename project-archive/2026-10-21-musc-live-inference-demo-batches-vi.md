# Kế hoạch chuẩn bị batch cho demo Live Inference MuSc

## Mục đích

Chuẩn bị các video demo quay sẵn cho chế độ Gradio Live Inference. Có thể giới
thiệu ứng dụng trong lúc trình bày và mở ứng dụng khi được yêu cầu, còn video
đóng vai trò phương án dự phòng đáng tin cậy nếu chạy trực tiếp trên Colab mất
quá nhiều thời gian hoặc gặp sự cố kết nối.

Demo đề xuất sử dụng hai dataset không thuộc các thí nghiệm tái lập trên MVTec
AD và VisA:

1. Một batch ảnh công nghiệp từ MPDD để khảo sát khả năng chuyển giao trong
   đúng miền ứng dụng dự kiến.
2. Một batch ảnh phi công nghiệp từ PlantVillage để quan sát hành vi của mô
   hình khi có domain shift.

Đây là các demo định tính, không phải đánh giá benchmark và không phải bằng
chứng về khả năng generalization.

## Quy tắc đầu vào chung

- Chạy từng dataset như một batch riêng. Không trộn ảnh công nghiệp và phi công
  nghiệp trong cùng một tập đầu vào MuSc.
- Sử dụng chính xác 40 ảnh thuộc cùng một semantic category và có điều kiện thu
  nhận tương đối nhất quán.
- Sử dụng 32 ảnh normal và 8 ảnh anomalous để ảnh bất thường vẫn là nhóm thiểu
  số trong batch.
- Đổi tên các file upload bằng định danh trung tính, không để lộ nhãn.
- Lưu riêng một bảng ánh xạ từ tên file trung tính đến đường dẫn nguồn và nhãn
  ground truth.
- Cố định danh sách ảnh và random seed trước khi quay vì điểm MuSc có tính
  transductive và có thể thay đổi khi thành phần batch thay đổi.
- Không so sánh trực tiếp giá trị anomaly score giữa hai batch. Điểm chỉ có ý
  nghĩa tương đối trong từng tập ảnh được upload.

## Batch A: Khảo sát chuyển giao trong miền công nghiệp

### Dataset và category

- Dataset: Metal Parts Defect Detection Dataset (MPDD)
- Category: `metal_plate`
- Mục tiêu quan sát: kiểm tra xem MuSc có tạo được thứ hạng hợp lý và heatmap
  hữu ích trên một dataset anomaly detection công nghiệp chưa xuất hiện trong
  các thí nghiệm tái lập hay không.

MPDD chứa hơn 1.000 ảnh linh kiện kim loại được sản xuất công nghiệp, bao gồm
mẫu không có lỗi, mẫu có lỗi và defect mask chính xác đến từng pixel.

### Thành phần đề xuất cho batch 40 ảnh

| Nhóm nguồn | Số lượng |
|---|---:|
| `good` | 32 |
| `scratches` | 3 |
| `major_rust` | 3 |
| `total_rust` | 2 |
| **Tổng cộng** | **40** |

Ưu tiên chọn ảnh normal từ test split hoặc validation split chính thức. Nếu
split đó không có đủ 32 ảnh normal phù hợp, có thể bổ sung mẫu từ `train/good`,
nhưng phải mô tả rõ đây là một custom demo batch, không phải một benchmark run
trên MPDD.

Tên file upload trung tính được đề xuất:

```text
mpdd_plate_001.png
mpdd_plate_002.png
...
mpdd_plate_040.png
```

### Giới hạn diễn giải

Kết quả hợp lý về mặt trực quan chỉ hỗ trợ phát biểu rằng MuSc có thể chạy trên
batch ảnh công nghiệp đã chọn. Một batch được tuyển chọn không phải kết quả
external validation và không được báo cáo như hiệu năng benchmark trên MPDD.

### Nguồn và giấy phép

- Dataset repository: <https://github.com/stepanje/MPDD>
- Bài báo liên quan: Stepan Jezek, Martin Jonak, Radim Burget, Pavel Dvorak, và
  Milos Skotak, "Deep learning-based defect detection of metal parts:
  evaluating current methods in complex conditions," ICUMT, 2021.
  <https://doi.org/10.1109/ICUMT54235.2021.9631567>
- Giấy phép: CC BY-NC-SA 4.0.
  <https://raw.githubusercontent.com/stepanje/MPDD/main/LICENSE>

## Batch B: Khảo sát domain shift ngoài miền công nghiệp

### Dataset và category

- Dataset: PlantVillage
- Loài cây: tomato
- Nhóm normal: `Tomato___healthy`
- Nhóm anomalous: `Tomato___Bacterial_spot`
- Mục tiêu quan sát: kiểm tra xem MuSc phản ứng với tổn thương trên lá, hình dạng
  lá hay nhiễu nền như thế nào khi được sử dụng ngoài miền kiểm định công
  nghiệp.

PlantVillage chứa hơn 54.000 ảnh lá khỏe mạnh và lá bị bệnh từ 14 loài cây. Sử
dụng cấu hình ảnh RGB gốc `color`, không sử dụng cấu hình segmented vì đường
biên nhân tạo có thể tạo ra phản ứng heatmap gây hiểu nhầm.

### Thành phần đề xuất cho batch 40 ảnh

| Nhóm nguồn | Số lượng |
|---|---:|
| `Tomato___healthy` | 32 |
| `Tomato___Bacterial_spot` | 8 |
| **Tổng cộng** | **40** |

Chọn 40 giá trị `leaf_id` khác nhau để tránh các góc chụp gần trùng của cùng
một chiếc lá. Ưu tiên sử dụng một split chính thức và cố định random seed.

Tên file upload trung tính được đề xuất:

```text
tomato_001.jpg
tomato_002.jpg
...
tomato_040.jpg
```

Có thể tải phiên bản trên Hugging Face bằng đoạn mã sau:

```python
from datasets import load_dataset

dataset = load_dataset("mohanty/PlantVillage", "color")
```

### Giới hạn diễn giải

PlantVillage chủ yếu là dataset cho image classification và không cung cấp
pixel-mask evaluation protocol giống MVTec AD, VisA hoặc MPDD. Chỉ xem kết quả
như một khảo sát định tính ngoài miền. Không mô tả anomaly score như xác suất
bệnh, chẩn đoán, hoặc bằng chứng cho thấy MuSc generalize sang bài toán phát
hiện bệnh trên cây trồng.

Lời dẫn được đề xuất:

> Đây là một khảo sát định tính ngoài miền nhằm quan sát hành vi của MuSc bên
> ngoài bài toán kiểm định công nghiệp. Đây không phải chẩn đoán bệnh trên cây
> và không chứng minh khả năng generalization.

### Nguồn và giấy phép

- Dataset repository: <https://github.com/spMohanty/PlantVillage-Dataset>
- Hugging Face dataset card: <https://huggingface.co/datasets/mohanty/PlantVillage>
- Bài báo liên quan: Sharada P. Mohanty, David P. Hughes, và Marcel Salathe,
  "Using Deep Learning for Image-Based Plant Disease Detection," Frontiers in
  Plant Science, 2016. <https://doi.org/10.3389/fpls.2016.01419>
- Package trên Hugging Face được tham chiếu xác định giấy phép là CC BY-SA 3.0.
  Giữ nguyên thông tin ghi công và tuân thủ các điều khoản của đúng bản phân
  phối được tải xuống.

## Kế hoạch quay video

Quay một video ngắn cho mỗi batch thay vì ghép cả hai lượt chạy thành một video
dài.

1. Hiển thị scope card và environment card.
2. Nói rõ đầu vào gồm 40 ảnh chưa gán nhãn thuộc cùng một category.
3. Upload batch ảnh đã được đổi sang tên trung tính.
4. Kiểm tra và xác nhận batch.
5. Bắt đầu Live Inference. Có thể tua nhanh thời gian chờ khi chỉnh sửa video
   nếu video ghi rõ đoạn chạy đã được rút ngắn.
6. Hiển thị bảng kết quả đã xếp hạng.
7. Kiểm tra một kết quả tốt, một ảnh normal có thứ hạng thấp và một heatmap
   chưa hoàn hảo hoặc có phản ứng lan rộng nếu có.
8. Tải kết quả CSV và ZIP.
9. Kết thúc bằng giới hạn phạm vi phù hợp với batch đang trình bày.

## Bằng chứng cần lưu lại

Với mỗi lượt quay, cần lưu:

- Manifest chính xác của 40 ảnh và bảng ánh xạ nhãn được giữ riêng.
- Nguồn dataset, giấy phép, category và seed dùng để chọn mẫu.
- Commit của reproduction repository và commit MuSc upstream.
- Phiên bản môi trường Colab và tên GPU.
- Thời gian khởi tạo model và thời gian inference.
- File CSV và ZIP đã tải xuống.
- Video hoàn chỉnh và một nhóm nhỏ screenshot dự phòng.
- Ghi chú về các heatmap thất bại hoặc gây hiểu nhầm, không che giấu các trường
  hợp này.

## Tóm tắt quyết định

- Batch công nghiệp được khuyến nghị: MPDD `metal_plate`.
- Batch phi công nghiệp được khuyến nghị: PlantVillage với lá cà chua khỏe mạnh
  và lá bị bacterial spot.
- Nếu chỉ đủ thời gian cho một batch ngoài benchmark, ưu tiên MPDD vì dataset
  này vẫn thuộc đúng miền industrial visual inspection của MuSc.
- Sử dụng video quay sẵn làm phương án dự phòng khi trình bày. Chỉ chạy ứng
  dụng trực tiếp trong buổi trình bày khi giảng viên yêu cầu và khi thời gian,
  kết nối Colab cho phép.
