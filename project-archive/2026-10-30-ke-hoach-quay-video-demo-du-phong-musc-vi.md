# MuSc Backup Demo Video Production Plan

> **Dành cho người thực hiện:** Theo dõi tiến độ bằng các checkbox trong tài liệu này. Mỗi video phải được kiểm tra độc lập trước khi đưa vào gói dự phòng.

**Mục tiêu:** Sản xuất hai video ngắn có lời thuyết minh để sử dụng khi demo trực tiếp MuSc trên Google Colab gặp sự cố.

**Cấu trúc:** Video MPDD trình bày một lượt Live Inference hoàn chỉnh trên batch công nghiệp hiện có. Video PlantVillage minh họa ranh giới phạm vi: app có thể kiểm tra cấu trúc file nhưng người trình bày không xác nhận sai rằng dữ liệu lá cây là một category sản phẩm công nghiệp, vì vậy không chạy inference.

**Công cụ:** Google Colab T4, app Gradio MuSc, công cụ quay màn hình macOS, micro, iMovie hoặc trình biên tập tương đương, xuất H.264/AAC 1080p ở 30 fps.

**Đặc tả nguồn:** `courses/CS2203/2026-10-21-musc-live-inference-demo-batches-vi.md`, `courses/CS2203/musc-demo/docs/musc_demo_presentation_checklist.md`, và storyboard được duyệt ngày 2026-10-30.

## Ràng buộc chung

- Video 1 sử dụng nguyên trạng `mpdd_demo_40.zip`: 26 ảnh `good`, 5 `scratches`, 5 `major_rust`, và 4 `total_rust`.
- Không phát biểu batch MPDD hiện tại có tỷ lệ 32/8, là benchmark MPDD, hoặc chứng minh khả năng generalization.
- Video 2 sử dụng `plantvillage_demo_40.zip`: 32 ảnh `Tomato___healthy` và 8 ảnh `Tomato___Bacterial_spot`, không có `leaf_id` trùng nhau.
- Không chọn ô xác nhận category sản phẩm công nghiệp và không chạy MuSc trên PlantVillage trong app hiện tại.
- Không hiển thị GitHub token, Colab Secrets, đường dẫn Drive cá nhân, email, thông báo riêng tư hoặc thanh địa chỉ chứa Gradio share URL.
- Lời thuyết minh phải mô tả đúng những gì đang xuất hiện trên màn hình. Không gọi anomaly score là xác suất, chẩn đoán hoặc nhãn lỗi chắc chắn.
- Hai video là sản phẩm độc lập. Video MPDD dài khoảng 2 phút 30 giây; video PlantVillage dài khoảng 1 phút 30 giây đến 2 phút.
- Giữ bản quay gốc đầy đủ. Nếu rút ngắn thời gian inference, chèn chữ `Đoạn xử lý đã được tua nhanh; kết quả không bị thay đổi.`

## Trọng tâm kiểm tra

- Runtime không phải T4 hoặc model chưa sẵn sàng: dừng quay, không che giấu trạng thái môi trường.
- Batch MPDD hiển thị số lượng khác 40 hoặc có lỗi file: không xác nhận và không chạy.
- Kết quả MPDD không phù hợp kỳ vọng: trình bày trung thực như đầu ra của model và chọn một heatmap chưa hoàn hảo để thảo luận.
- PlantVillage chỉ vượt qua kiểm tra cấu trúc file: giải thích rõ app không tự xác minh domain, không coi đó là chấp thuận chạy.
- Video làm lộ thông tin riêng tư hoặc âm thanh khó nghe: quay lại cảnh bị ảnh hưởng thay vì cố che bằng lời giải thích.

---

## Tệp đầu vào và đầu ra

### Đầu vào đã có

- `mpdd_demo_40.zip`
- `mpdd_manifest.csv`
- `plantvillage_demo_40.zip`
- `plantvillage_manifest.csv`
- `musc-classroom-prepared-bundle.zip`
- `notebooks/MuSc_classroom_demo_vi.ipynb`

Manifest chỉ dùng để hậu kiểm. Không upload manifest cùng batch và không cho model đọc nhãn.

### Đầu ra cần tạo

- `CS2203_MuSc_MPDD_live_inference_backup_1080p.mp4`
- `CS2203_MuSc_PlantVillage_scope_boundary_1080p.mp4`
- Bản quay gốc của từng video.
- CSV và ZIP do lượt chạy MPDD thành công tạo ra.
- Ba ảnh chụp MPDD: environment card, bảng xếp hạng, và ví dụ heatmap/overlay.
- Một ghi chú ngắn chứa ngày quay, GPU, reproduction commit, MuSc commit, thời gian khởi tạo model và thời gian inference.

## Task 1: Chuẩn bị phiên quay sạch

**Kết quả kiểm tra được:** Một runtime mới, model sẵn sàng, app mở ở chế độ toàn màn hình và thu thử có hình tiếng rõ ràng.

- [ ] Bật chế độ không làm phiền, đóng email và ứng dụng nhắn tin, ẩn bookmark cá nhân và tắt tự động điền mật khẩu.
- [ ] Cắm nguồn, kiểm tra kết nối mạng ổn định và ngăn máy tự chuyển sang chế độ ngủ.
- [ ] Mở `MuSc_classroom_demo_vi.ipynb` trong một Colab runtime mới có T4 GPU và chạy các cell thiết lập theo đúng thứ tự.
- [ ] Kiểm tra reproduction commit được in bởi runtime và MuSc upstream commit là `b76b93da8bd3096a99964a96ae29d46f197a0651`.
- [ ] Mount prepared bundle, mở app và nhấn `Khởi tạo MuSc model` trước khi bắt đầu quay nội dung chính.
- [ ] Xác nhận environment card hiển thị CUDA GPU và model ở trạng thái sẵn sàng.
- [ ] Mở app ở cửa sổ riêng, ẩn thanh địa chỉ hoặc crop khỏi vùng quay, đặt zoom sao cho scope card, environment card và nút thao tác đọc được ở video 1080p.
- [ ] Chọn micro đúng, đặt máy cách miệng khoảng 15-20 cm và thu thử 10 giây.
- [ ] Nghe lại bản thử bằng tai nghe, xác nhận không rè, không vọng, lời nói rõ và con trỏ chuột dễ nhìn.

## Task 2: Quay Video 1 - MPDD Live Inference

**Kết quả kiểm tra được:** Một bản quay liên tục chứa lượt chạy MPDD thành công, kết quả xếp hạng, ví dụ định tính và tệp tải xuống.

### Shot 1 - Mở đầu và phạm vi, 0:00-0:20

- [ ] Bắt đầu ở đầu trang để thấy tiêu đề, scope card và environment card.
- [ ] Đọc lời dẫn:

  > Đây là ứng dụng minh họa MuSc cho phát hiện và định vị bất thường ngoại quan. MuSc không cần huấn luyện riêng cho category đang thử. Đầu ra là anomaly score tương đối trong chính batch được tải lên, kèm heatmap và overlay, không phải nhãn lỗi chắc chắn.

### Shot 2 - Giới thiệu batch, 0:20-0:45

- [ ] Chuyển sang tab `Thử nghiệm tập ảnh mới` và dừng con trỏ ở hợp đồng input.
- [ ] Đọc lời dẫn:

  > Batch này gồm 40 ảnh của cùng category metal plate từ MPDD. Tên file đã được trung tính hóa và app không nhận ground-truth label. Batch hiện có 26 ảnh good và 14 ảnh thuộc ba nhóm lỗi, nên đây là một batch demo tùy chỉnh chứ không phải quy trình benchmark MPDD.

- [ ] Upload duy nhất `mpdd_demo_40.zip`.
- [ ] Đợi validation report hiển thị đủ 40 ảnh hợp lệ.

### Shot 3 - Xác nhận và chạy, 0:45-1:10

- [ ] Chọn ô xác nhận vì MPDD `metal_plate` đúng là một category sản phẩm công nghiệp có điều kiện chụp tương đối nhất quán.
- [ ] Đọc lời dẫn:

  > App chỉ kiểm tra file và số lượng. Việc xác nhận category và domain vẫn thuộc trách nhiệm của người trình bày. Điểm của từng ảnh phụ thuộc vào thành phần của toàn batch và không nên so sánh trực tiếp với batch khác.

- [ ] Nhấn `Chạy MuSc live inference` và giữ nguyên bản quay cho đến khi kết quả xuất hiện.
- [ ] Ghi lại thời gian inference thực tế. Không chạy lại chỉ để tìm kết quả đẹp hơn.

### Shot 4 - Kết quả, 1:10-2:10 sau khi dựng

- [ ] Cho thấy tiêu đề bảng `Xếp hạng bằng anomaly score tương đối` và cuộn chậm qua một phần gallery.
- [ ] Mở ảnh có adjusted score cao nhất và đọc:

  > Ảnh này được model xếp hạng đáng ngờ nhất trong batch. Heatmap cho biết những vùng đóng góp mạnh vào điểm số, nhưng không tự chứng minh đây là lỗi thật.

- [ ] Mở một ảnh trong năm hạng thấp nhất và đọc:

  > Ở đầu còn lại của bảng xếp hạng, ảnh này có mức độ bất thường tương đối thấp hơn so với các ảnh còn lại trong cùng batch.

- [ ] Chọn một heatmap có phản ứng lan sang biên vật thể, nền hoặc vùng không thuyết phục và đọc:

  > Trường hợp này cho thấy một giới hạn quan trọng: heatmap có thể phản ứng với biên, nền hoặc biến thiên hình ảnh, nên cần được kiểm tra định tính thay vì diễn giải như ground truth.

- [ ] Sau khi dừng quay, dùng `mpdd_manifest.csv` để hậu kiểm ba file vừa trình bày và ghi lại tên file cùng nhãn thật trong ghi chú nội bộ. Không sửa lời mô tả model dựa trên thông tin mà model không được nhận.

### Shot 5 - Lưu đầu ra và kết luận, 2:10-2:30

- [ ] Bấm `Tải CSV` và `Tải ZIP kết quả`.
- [ ] Đọc lời kết:

  > CSV và ZIP giúp lưu lại thứ hạng, heatmap và overlay để kiểm tra sau. Lượt chạy này chỉ cho thấy MuSc có thể xử lý batch MPDD đã chọn. Nó không phải kết quả benchmark trên toàn MPDD và không chứng minh khả năng generalization.

## Task 3: Quay Video 2 - PlantVillage Scope Boundary

**Kết quả kiểm tra được:** Một video ngắn cho thấy app kiểm tra được cấu trúc batch nhưng người trình bày chủ động dừng vì dữ liệu nằm ngoài phạm vi công nghiệp.

### Shot 1 - Nêu mục đích, 0:00-0:20

- [ ] Bắt đầu ở scope card rồi chuyển sang tab `Thử nghiệm tập ảnh mới`.
- [ ] Đọc lời dẫn:

  > Video này kiểm tra cách xử lý một batch nằm ngoài phạm vi đã công bố. PlantVillage chứa ảnh lá cây và không phải dữ liệu kiểm tra bất thường ngoại quan trong công nghiệp.

### Shot 2 - Upload và đọc validation, 0:20-0:55

- [ ] Upload duy nhất `plantvillage_demo_40.zip`.
- [ ] Đợi validation report hiển thị đủ 40 ảnh hợp lệ.
- [ ] Đọc lời dẫn:

  > Batch có 40 ảnh lá cà chua với tên file trung tính. Kiểm tra này chỉ xác nhận định dạng, số lượng và tính hợp lệ của file. App không tự biết đây là ảnh lá cây và cũng không tự xác minh domain.

### Shot 3 - Không xác nhận sai, 0:55-1:25

- [ ] Giữ ô xác nhận category sản phẩm công nghiệp ở trạng thái bỏ chọn.
- [ ] Dừng con trỏ ở ô xác nhận và trạng thái nút chạy.
- [ ] Đọc lời dẫn:

  > Để chạy, người dùng phải xác nhận batch thuộc một category sản phẩm công nghiệp với điều kiện chụp tương đối nhất quán. PlantVillage không đáp ứng điều kiện này, nên tôi không chọn ô xác nhận và không chạy model. Việc file hợp lệ không đồng nghĩa với việc bài toán nằm trong phạm vi sử dụng.

### Shot 4 - Kết luận, 1:25-1:45

- [ ] Trở lại scope card và đọc:

  > Muốn nghiên cứu domain shift sang bệnh trên cây cần một protocol riêng, nhãn, metric, baseline và đánh giá phù hợp. Video này minh họa ranh giới sử dụng có trách nhiệm, không phải một kết quả inference hay bằng chứng generalization.

## Task 4: Dựng hai video

**Kết quả kiểm tra được:** Hai tệp MP4 đúng thời lượng, đọc được nội dung giao diện, nghe rõ lời nói và không làm thay đổi ý nghĩa của lượt chạy.

- [ ] Cắt khoảng im lặng ở đầu và cuối nhưng giữ nguyên thứ tự thao tác.
- [ ] Với MPDD, rút ngắn đoạn inference bằng tăng tốc thay vì xóa hoàn toàn chuyển tiếp từ lúc bấm chạy đến lúc có kết quả.
- [ ] Chèn chữ `Đoạn xử lý đã được tua nhanh; kết quả không bị thay đổi.` trong toàn bộ đoạn tăng tốc.
- [ ] Không tua nhanh lời thuyết minh. Thu bổ sung voice-over nếu lời nói trong bản quay gốc bị vấp hoặc khó nghe.
- [ ] Chèn tiêu đề đầu video MPDD: `MuSc Live Inference - MPDD metal_plate - Demo định tính`.
- [ ] Chèn tiêu đề đầu video PlantVillage: `MuSc - Minh họa ranh giới phạm vi với PlantVillage`.
- [ ] Thêm phụ đề tiếng Việt cho các câu kết luận và các giới hạn quan trọng.
- [ ] Giữ con trỏ và nội dung chính trong safe area, không crop mất nhãn nút, validation report hoặc tên cột bảng kết quả.
- [ ] Xuất 1920 x 1080, 30 fps, H.264 video và AAC audio.

## Task 5: Kiểm tra khoa học và kỹ thuật

**Kết quả kiểm tra được:** Hai video vượt qua toàn bộ checklist và có thể phát độc lập mà không cần giải thích bổ sung để sửa sai nội dung.

### Video MPDD

- [ ] Thời lượng nằm trong khoảng 2:00-3:00.
- [ ] Environment card cho thấy model sẵn sàng trước khi chạy.
- [ ] Validation report cho thấy đúng 40 ảnh.
- [ ] Lời thoại nói đúng 26 `good` và 14 ảnh lỗi, không nói 32/8.
- [ ] Không có phát biểu `benchmark`, `độ chính xác`, `xác suất lỗi` hoặc `chứng minh generalization` theo nghĩa khẳng định.
- [ ] Bảng xếp hạng, ít nhất ba ví dụ định tính, nút tải CSV và nút tải ZIP đều xuất hiện.
- [ ] CSV và ZIP tải xuống mở được; PNG trong ZIP hiển thị được.

### Video PlantVillage

- [ ] Thời lượng nằm trong khoảng 1:20-2:00.
- [ ] Validation report cho thấy đúng 40 ảnh.
- [ ] Ô xác nhận công nghiệp không được chọn và nút inference không được bấm.
- [ ] Lời thoại phân biệt rõ kiểm tra cấu trúc file với xác minh domain.
- [ ] Video không chứa anomaly score, heatmap hoặc kết luận bệnh cây do MuSc tạo ra.

### Cả hai video

- [ ] Xem toàn bộ video từ đầu đến cuối bằng một trình phát khác với trình biên tập.
- [ ] Kiểm tra bằng tai nghe và loa laptop; giọng nói phải rõ ở cả hai.
- [ ] Tạm dừng tại mỗi cảnh có chữ nhỏ để xác nhận nội dung đọc được ở 1080p.
- [ ] Không có token, email, thông báo riêng tư, đường dẫn Drive hoặc Gradio share URL trong bất kỳ frame nào.
- [ ] Tên video, tiêu đề, lời thuyết minh và nội dung app thống nhất với nhau.

## Task 6: Đóng gói phương án dự phòng

**Kết quả kiểm tra được:** Một thư mục có thể mở ngoại tuyến, chứa video chính, đầu ra MPDD và ảnh chụp cần thiết.

- [ ] Tạo thư mục `CS2203_MuSc_demo_backup_2026-10-30` ở nơi lưu trữ cá nhân phù hợp.
- [ ] Chép hai video MP4 hoàn chỉnh vào thư mục.
- [ ] Chép CSV, ZIP kết quả và ba ảnh chụp từ lượt chạy MPDD thành công.
- [ ] Chép ghi chú môi trường và thời gian chạy, không chứa token hoặc đường dẫn riêng tư.
- [ ] Mở từng tệp trực tiếp từ thư mục dự phòng để kiểm tra lần cuối.
- [ ] Chép thêm một bản của thư mục sang thiết bị hoặc vị trí lưu trữ ngoại tuyến thứ hai.

## Thứ tự phát khi thuyết trình

1. Nếu demo trực tiếp hoạt động, chỉ dùng video khi thời gian chờ quá lâu hoặc kết nối thất bại.
2. Phát video MPDD trước vì đây là ví dụ nằm trong đúng miền ứng dụng công nghiệp.
3. Chỉ phát video PlantVillage khi cần giải thích giới hạn phạm vi, validation boundary hoặc câu hỏi về dữ liệu ngoài miền.
4. Nếu chỉ có thời gian cho một video, chọn MPDD.
