# Dàn ý trình bày đồ án MuSc với giảng viên

## 1. Mục tiêu buổi trình bày

Buổi trình bày cần giúp người nghe hiểu ba thông điệp chính:

1. MuSc giải quyết đồng thời phát hiện bất thường cấp ảnh và định vị vùng lỗi cấp pixel mà không huấn luyện trên dataset đích và không dùng text prompt.
2. Đồ án đã tái lập MuSc trên toàn bộ MVTec AD và VisA; sai lệch pixel mean lớn nhất so với baseline repository là 0,0044 điểm phần trăm.
3. Kết quả cần được đọc cùng giới hạn quan sát được: RsCIN có delta khác nhau theo lớp, AUROC cao không đảm bảo định vị chặt, và top 1% pooling cho kết quả tổng hợp thấp hơn maximum pooling trên MVTec AD.

## 2. Nguyên tắc trình bày

- Không đọc nguyên văn nội dung trên slide.
- Với bảng số liệu, chỉ nêu một hoặc hai con số quan trọng nhất.
- Luôn phân biệt rõ cấp ảnh và cấp pixel.
- Nói "điểm phần trăm" khi mô tả chênh lệch giữa các metric được biểu diễn theo phần trăm.
- Không tuyên bố nguyên nhân khi mới chỉ có giả thuyết. Ví dụ, RsCIN và khác biệt môi trường là giả thuyết cho sai lệch cấp ảnh, chưa phải nguyên nhân đã được chứng minh.
- Khi thiếu thời gian, ưu tiên câu chuyện chính: tầm quan trọng, phát biểu bài toán, ý tưởng MuSc, dataset, thiết kế tái lập, kết quả, failure cases và kết luận.

Quy ước mức ưu tiên:

- **Bắt buộc:** phải nói để câu chuyện không bị đứt.
- **Nên nói:** bổ sung bằng chứng hoặc giải thích quan trọng.
- **Có thể lược:** nói nhanh hoặc bỏ nếu thời gian hạn chế.

## 3. Dàn ý theo từng slide

### Slide 1 - Trang tiêu đề

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Giới thiệu đề tài, phạm vi và người thực hiện.

**Ý cần nói:**

- Em trình bày đồ án tái hiện phương pháp MuSc cho phát hiện và phân vùng bất thường zero-shot trên ảnh công nghiệp.
- Đồ án tập trung vào hai vấn đề: tái lập kết quả và kiểm tra giới hạn thực nghiệm của phương pháp.

**Không cần đọc:** Tên trường, mã môn và toàn bộ dòng phụ đề.

**Câu chuyển:** "Trước hết, em xin bắt đầu từ lý do bài toán này khó trong môi trường công nghiệp."

### Slide 2 - Bối cảnh bài toán kiểm định công nghiệp

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Giải thích tầm quan trọng và khó khăn của bài toán.

**Ý cần nói:**

- Ảnh lỗi hiếm, loại lỗi mới khó dự đoán trước và mask theo pixel tốn chi phí.
- Phát hiện sớm giúp hạn chế sản phẩm lỗi lọt qua kiểm định.
- Tuy nhiên, không thể thu thập và gán nhãn trước mọi kiểu lỗi có thể xuất hiện.
- Hình `leather/glue/011` minh họa chuỗi ảnh gốc, ground truth chỉ dùng khi đánh giá, heatmap và overlay; không mô tả ground truth như input của MuSc.

**Có thể lược:** Không cần giải thích riêng từng ví dụ về thay đổi ánh sáng hoặc vật liệu.

**Câu chuyển:** "Từ nhu cầu đó, em phát biểu chính xác input, output và phạm vi của bài toán MuSc."

### Slide 3 - Bài toán và đầu ra của MuSc

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Rào rõ input, output, giả định và những trường hợp ngoài protocol trước khi trình bày phương pháp.

**Ý cần nói:**

- Input là một tập $D_c=\{x_1,\ldots,x_N\}$ gồm nhiều ảnh test chưa gán nhãn thuộc cùng một loại sản phẩm (product category) $c$.
- Tập có thể chứa cả ảnh normal và anomaly; label, mask và text prompt không được dùng khi inference.
- Với mỗi ảnh, output gồm anomaly score cấp ảnh và anomaly map cấp pixel để định vị vùng lỗi.
- Nhãn normal/anomaly chỉ được tạo sau khi chọn threshold; output trực tiếp của MuSc là score và map.
- Ví dụ, một tập $N$ ảnh `bottle` tạo ra $N$ anomaly score và $N$ heatmap.
- MuSc giả định pattern normal lặp lại giữa nhiều ảnh, còn anomaly tương đối hiếm.
- Một ảnh cô lập hoặc một tập trộn nhiều loại sản phẩm không phải input đúng theo protocol gốc.

**Cách trình bày sơ đồ:** Chỉ từ trái sang phải: tập ảnh cùng category, khối MuSc, rồi hai đầu ra score và map. Sau đó mới chỉ xuống dải đỏ để rào label, mask và text prompt không được dùng khi inference.

**Câu cần rào trước:** "Category mới vẫn xử lý được nếu có một tập nhiều ảnh cùng category; một ảnh duy nhất thì không đủ cho mutual scoring theo protocol gốc."

**Câu chuyển:** "Với protocol này, MuSc khác các phương pháp cần ảnh normal tham chiếu hoặc text prompt như thế nào?"

### Slide 4 - MuSc trong các protocol anomaly detection

**Mức ưu tiên:** Nên nói

**Mục tiêu:** Xác định đúng protocol và điểm khác biệt của MuSc.

**Ý cần nói:**

- PatchCore cần ảnh normal của lớp đích để xây dựng memory bank.
- WinCLIP không cần ảnh train của lớp đích nhưng sử dụng text prompt.
- AnomalyCLIP học prompt trên dữ liệu phụ trợ.
- MuSc không dùng ảnh train của lớp đích, không anomaly-specific training và không dùng text prompt khi inference.
- Đánh đổi là MuSc có tính transductive: dự đoán một ảnh phụ thuộc vào các ảnh khác trong test set.

**Điểm cần tránh:** Không nói bảng này là bảng xếp hạng hiệu năng. Các phương pháp có protocol khác nhau.

**Câu chuyển:** "Điểm khác biệt đó xuất phát từ cách MuSc dùng chính test set làm nguồn tham chiếu."

### Slide 5 - Pipeline MuSc

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Cho người nghe thấy toàn bộ luồng xử lý trước khi đi vào từng thành phần.

**Ý cần nói:**

1. CLIP ViT trích xuất patch token và class token.
2. LNAMD tạo đặc trưng patch ở nhiều mức ngữ cảnh.
3. MSM so khớp patch giữa các ảnh trong cùng lớp sản phẩm.
4. MSM tạo anomaly map cấp pixel.
5. Maximum pooling tạo score cấp ảnh ban đầu.
6. RsCIN hiệu chỉnh score cấp ảnh dựa trên các ảnh có hình dạng tương tự.

**Thông điệp trọng tâm:** Patch normal thường lặp lại giữa nhiều ảnh, còn patch bất thường hiếm nên khó tìm được đối sánh gần.

**Câu chuyển:** "Ba thành phần quan trọng nhất của pipeline là LNAMD, MSM và RsCIN."

### Slide 6 - LNAMD: biểu diễn patch ở nhiều mức ngữ cảnh

**Mức ưu tiên:** Nên nói

**Mục tiêu:** Giải thích vì sao MuSc dùng nhiều mức lân cận.

**Ý cần nói:**

- `r = 1` giữ chi tiết cục bộ.
- `r = 3` và `r = 5` bổ sung ngữ cảnh rộng hơn.
- MuSc giữ cả ba biểu diễn vì khuyết tật có thể có nhiều kích thước khác nhau.

**Có thể lược:** Không cần mô tả thao tác average pooling quá chi tiết nếu thầy không hỏi.

**Câu chuyển:** "Sau khi có đặc trưng đa mức, MSM quyết định patch nào là hiếm so với các ảnh còn lại."

### Slide 7 - MSM: tính điểm từ đối sánh patch liên ảnh

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Giải thích cơ chế cốt lõi của MuSc.

**Ý cần nói:**

- Với một patch, MuSc tìm patch gần nhất trong từng ảnh khác.
- Các khoảng cách liên ảnh được sắp xếp và chỉ lấy trung bình 30% nhỏ nhất.
- Patch normal có nhiều đối sánh gần nên nhận điểm thấp.
- Patch bất thường ít có đối sánh gần nên nhận điểm cao.
- Chỉ lấy 30% nhỏ nhất giúp một patch normal không cần giống toàn bộ test set, chỉ cần giống một nhóm ảnh đủ lớn.

**Không cần đọc:** Không cần đọc từng ký hiệu trong công thức. Chỉ chỉ vào công thức và giải thích ý nghĩa.

**Câu chuyển:** "Anomaly map đã được tạo ở đây, nhưng score cấp ảnh còn được hiệu chỉnh thêm bằng RsCIN."

### Slide 8 - RsCIN: hiệu chỉnh score cấp ảnh

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Phân biệt vai trò của RsCIN với MSM và anomaly map.

**Ý cần nói:**

- Maximum pooling nhạy với một điểm nóng cô lập trên ảnh normal.
- RsCIN dùng cosine similarity giữa class token để tìm các ảnh có hình dạng tổng thể tương tự.
- Score cấp ảnh được kết hợp trong constrained neighborhood.
- RsCIN chỉ thay đổi classification score, không thay đổi anomaly map và metric cấp pixel.

**Thông điệp cần nhấn:** Giả định "ảnh giống nhau nên có mức bất thường tương tự" cần được kiểm tra, không phải lúc nào cũng đúng.

**Câu chuyển:** "Tiếp theo là cách em khóa cấu hình để kiểm tra khả năng tái lập trên hai benchmark."

### Slide 9 - Dataset, metric và protocol tái lập

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Giải thích dữ liệu cụ thể của bài toán công nghiệp và chứng minh thực nghiệm được thiết kế có kiểm soát, có thể audit.

**Ý cần nói:**

- MVTec AD là benchmark kiểm định công nghiệp gồm 5.354 ảnh độ phân giải cao thuộc 15 category: 10 object và 5 texture, chẳng hạn bottle, cable, leather và carpet. Test set có ảnh normal, ảnh defective và pixel mask cho vùng lỗi.
- VisA là Visual Anomaly Dataset gồm 10.821 ảnh thuộc 12 loại vật thể trong ba nhóm: PCB có cấu trúc phức tạp, cảnh có nhiều vật thể, và vật thể gần căn chỉnh. Dataset có 9.621 ảnh normal và 1.200 ảnh anomaly.
- Đồ án chạy toàn bộ 15 category MVTec AD và 12 category VisA, tổng cộng 27 category. Phần VisA của protocol sử dụng 2.162 ảnh test, không phải toàn bộ 10.821 ảnh của dataset.
- Label và mask ground truth chỉ được dùng sau inference để tính metric, không được dùng làm input cho MuSc.
- Giữ một cấu hình MuSc tham chiếu: CLIP ViT-L/14@336, input 518 x 518, bốn feature layer, LNAMD `[1, 3, 5]` và MSM 30%.
- Khóa commit `b76b93d`, lưu log, config, command và kết quả theo từng lớp.
- Mỗi lớp chạy trong subprocess riêng để có thể resume sau khi Colab bị ngắt.
- Metric cấp ảnh gồm AUROC, AP và F1-max, dùng để đánh giá xếp hạng và phân loại ảnh bất thường.
- Metric cấp pixel có thêm PRO để bổ sung đánh giá mức độ bao phủ theo vùng lỗi.

**Có thể lược:** Phiên bản chi tiết của Python và PyTorch có thể chỉ nhắc là môi trường Colab mới hơn môi trường chính thức.

**Câu chuyển:** "Với protocol đã khóa, em đối chiếu E01 với baseline repository trên MVTec AD."

### Slide 10 - Đối chiếu kết quả trên MVTec AD

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Trình bày bằng chứng tái lập chính trên benchmark đầu tiên.

**Ý cần nói:**

- Bảng dùng kết quả official repository tại commit `b76b93d` làm baseline exact vì E01 chạy cùng code và cấu hình đã ghim. Bảng trong paper là mốc công bố của một lần chạy official khác nên có thể lệch nhẹ.
- Hoàn thành 15 trên 15 lớp.
- Sai lệch tuyệt đối lớn nhất của các giá trị mean chỉ khoảng 0,036 điểm phần trăm.
- Toàn bộ 60 metric cấp pixel theo lớp lệch dưới 0,10 điểm phần trăm.
- Nhóm texture cao hơn object trên cả bảy metric trong phép so sánh mô tả, lớn nhất là 6,3305 điểm ở Pixel AP.

**Điểm cần tránh:** Không kết luận texture vốn dễ hơn object vì số lớp, số ảnh và kiểu lỗi khác nhau.

**Câu chuyển:** "Tiếp theo là phép đối chiếu tương tự trên VisA, nơi sai lệch tập trung nhiều hơn ở cấp ảnh."

### Slide 11 - Đối chiếu kết quả trên VisA

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Chứng minh tái lập trên dataset thứ hai và trình bày sai lệch trung thực.

**Ý cần nói:**

- E02 tiếp tục được đối chiếu với official repository tại commit `b76b93d`, không trộn với kết quả từ pipeline hoặc protocol khác.
- Hoàn thành 12 trên 12 lớp với 2.162 ảnh test.
- Bốn metric cấp pixel mean lệch dưới 0,004 điểm phần trăm.
- Metric cấp ảnh thấp hơn repository từ khoảng 0,13 đến 0,26 điểm phần trăm.
- Audit phát hiện ô Pixel AUROC của `fryum` trong bảng repository không nhất quán với mean công bố.
- Không tự sửa bảng tham chiếu, chỉ báo cáo giá trị in ra và giá trị suy ra từ mean.

**Điểm cần tránh:** Không khẳng định chắc chắn RsCIN gây ra toàn bộ sai lệch cấp ảnh.

**Câu chuyển:** "Các bảng cho biết mức độ khớp, nhưng heatmap mới cho biết mô hình đang tập trung vào đâu."

### Slide 12 - Ví dụ anomaly map phù hợp vùng lỗi

**Mức ưu tiên:** Nên nói

**Mục tiêu:** Minh họa điều kiện giúp mutual scoring hoạt động tốt.

**Ý cần nói:**

- `leather/glue/011` có heatmap gọn, đỉnh nằm trong mask, Pixel AUROC theo ảnh 0,9999.
- `pcb4/Anomaly/032` có đỉnh cục bộ trùng lỗi nhỏ trên connector, Pixel AUROC theo ảnh 0,9997.
- Điểm chung là patch lỗi nhỏ, nổi bật và đủ khác biệt với cấu trúc normal lặp lại.
- Pixel AUROC ở slide này tính trên từng ảnh, khác metric cấp category ở slide 10-11.
- Ground truth được hiển thị nhị phân vì mask VisA lưu class ID (1-7), không phải giá trị 255 như MVTec AD.

**Không cần đọc:** Không cần mô tả lại từng ô Original, Ground truth, Heatmap và Overlay nếu hình đã rõ.

**Câu chuyển:** "Tuy nhiên, metric cao không có nghĩa là mọi anomaly map đều định vị chặt."

### Slide 13 - Ví dụ anomaly map chưa bám sát vùng lỗi

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Cho thấy giới hạn mà số liệu tổng hợp có thể che khuất.

**Ý cần nói:**

- `transistor/cut_lead/008` có đỉnh heatmap nằm đúng tại chân bị cắt, nhưng phần lớn mask là đoạn chân bị thiếu, trông giống nền và không nổi bật, nên Pixel AUROC theo ảnh chỉ 0,5957.
- `fryum/Anomaly/010` đạt Pixel AUROC theo ảnh 0,9250, thấp nhất trong 100 ảnh lỗi của `fryum`. Đáp ứng mạnh nhất nằm ở nền bên trái, cách lỗi hơn 100 pixel trên ảnh 518 x 518; lỗi mảnh ở biên yếu hơn.
- Vì vậy cần đọc AUROC cùng AP, PRO, heatmap và hình dạng ground truth.

**Điểm cần tránh:** Lý do đoạn chân bị thiếu không nổi bật mới là quan sát từ hình, chưa phải nguyên nhân đã được chứng minh.

**Câu chuyển:** "Ngoài failure case cấp pixel, em còn kiểm tra riêng ảnh hưởng của RsCIN ở cấp ảnh."

### Slide 14 - Ảnh hưởng của RsCIN trên bốn lớp VisA

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Mô tả hướng và độ lớn của delta sau RsCIN trên bốn lớp VisA đã phân tích.

**Ý cần nói:**

- Trước tiên chỉ đọc hướng của thanh: bên trái mốc 0 là metric giảm sau RsCIN, bên phải là metric cải thiện. Ba màu lần lượt là AUROC, F1-max và AP.
- Trên bốn lớp VisA được phân tích, RsCIN cải thiện `pcb2`, `pcb4` và `fryum`.
- `pcb4` tăng hơn 10 điểm F1-max.
- `pcb3` giảm cả AUROC, F1-max và AP.
- Trên `pcb3`, 85 trên 100 ảnh bất thường bị giảm score, trong khi 72 trên 101 ảnh normal tăng score.
- Image-level similarity không đồng nghĩa với anomaly-level similarity.

**Cách trình bày biểu đồ:** Không đọc đủ 12 con số. Chỉ vào ba nhóm thanh nằm bên phải để nói ba lớp được cải thiện, sau đó chỉ vào ba thanh của `pcb3` nằm bên trái mốc 0 để nêu phản ví dụ. Chốt bằng hai con số dễ nhớ: `pcb4` tăng 10,13 điểm F1-max; `pcb3` giảm khoảng 1,3 đến 2,0 điểm trên cả ba metric.

**Lời nói gợi ý:** "Biểu đồ này chỉ cần đọc theo hướng. Thanh sang phải nghĩa là RsCIN cải thiện metric, còn thanh sang trái nghĩa là metric giảm. Ba trong bốn lớp được cải thiện, nổi bật là `pcb4` tăng hơn 10 điểm F1-max. Riêng `pcb3` giảm trên cả ba metric; đồng thời đa số ảnh anomaly bị kéo score xuống và đa số ảnh normal bị kéo score lên. Vì vậy, lân cận giống nhau về hình dạng ảnh chưa chắc giống nhau về mức bất thường."

**Điểm cần tránh:** Đây là phân tích trên bốn lớp được chọn, không phải kết luận rằng RsCIN gây hại trên toàn VisA.

**Câu chuyển:** "Từ điểm yếu của maximum pooling, em kiểm tra một thay đổi nhỏ là lấy trung bình top 1% pixel."

### Slide 15 - So sánh Top 1% pooling và maximum pooling

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Trình bày controlled extension và giữ nguyên kết quả âm.

**Ý cần nói:**

- Maximum và top 1% được tính từ cùng anomaly map, nên chỉ thay đổi phép tổng hợp score cấp ảnh.
- Sau RsCIN, top 1% làm giảm trung bình 15 lớp: AUROC 0,9529, F1-max 0,4310 và AP 0,2762 điểm phần trăm.
- Đây là mean, không phải lớp nào cũng giảm: với AUROC có 5 lớp tăng, 4 lớp hòa, 6 lớp giảm; median delta bằng 0 và riêng `screw` giảm 9,7151 điểm.
- `carpet/good/019` là ca có lợi: top 1% sửa false positive do đỉnh cục bộ.
- `screw/good/003` là phản ví dụ: đáp ứng trải gần toàn vật thể nên top 1% tạo false positive mới.
- Kết luận là top 1% có thể sửa một số ca riêng lẻ nhưng không thay thế maximum pooling trên toàn bộ 15 lớp MVTec AD.

**Điểm cần nhấn:** Không điều chỉnh tỷ lệ pooling sau khi thấy kết quả. Kết quả âm vẫn được giữ và báo cáo.

**Câu chuyển:** "Từ toàn bộ kết quả trên, em rút ra ba kết luận chính."

### Slide 16 - Kết luận và giới hạn thực nghiệm

**Mức ưu tiên:** Bắt buộc

**Mục tiêu:** Kết thúc bằng câu trả lời trực tiếp cho mục tiêu đồ án.

**Ý cần nói:**

1. Đã hoàn thành 27 trên 27 category; sai lệch pixel mean lớn nhất so với baseline repository là 0,0044 điểm phần trăm.
2. Anomaly map có thể bám vùng lỗi nhỏ, nổi bật hoặc lan sang cấu trúc lân cận; AUROC cần được đọc cùng AP, PRO và hình định tính.
3. RsCIN có delta khác nhau theo lớp; Top 1% pooling cho kết quả tổng hợp thấp hơn maximum trên MVTec AD; diễn giải kết quả cần đi cùng protocol và failure analysis.

**Câu kết đề xuất:** "Phép tái lập cho sai lệch nhỏ ở các metric cấp pixel, đồng thời cho thấy metric tổng hợp cần được đọc cùng protocol, cơ chế và các trường hợp định tính."

**Câu chuyển:** "Slide cuối là các tài liệu tham khảo chính của đồ án."

### Slide 17 - Tài liệu tham khảo chính

**Mức ưu tiên:** Có thể lược

**Mục tiêu:** Ghi nhận nguồn và kết thúc phần trình bày.

**Ý cần nói:**

- Công trình chính là MuSc tại ICLR 2024.
- Các tài liệu còn lại cung cấp baseline liên quan, backbone và hai benchmark.
- Cảm ơn thầy và mời thầy đặt câu hỏi.

**Không cần đọc:** Không đọc từng tài liệu tham khảo.

## 4. Phiên bản rút gọn khi thiếu thời gian

Nếu thời gian bị rút ngắn, ưu tiên theo thứ tự:

1. Slide 2 - Tầm quan trọng của bài toán.
2. Slide 3 - Input, output, ví dụ và phạm vi.
3. Slide 5 - Pipeline tổng thể.
4. Slide 7 - MSM, cơ chế cốt lõi.
5. Slide 9 - Dataset và thiết kế tái lập.
6. Slide 10 và 11 - Kết quả MVTec AD và VisA.
7. Slide 13 - Failure cases.
8. Slide 15 - Thử nghiệm top 1% và kết quả âm.
9. Slide 16 - Kết luận.

Có thể nói nhanh slide 4, 6, 8, 12 và 14. Slide 17 chỉ cần dừng lại để nhận câu hỏi.

## 5. Câu hỏi giảng viên có thể hỏi

### 5.1. MuSc có thực sự là zero-shot không?

Có, theo nghĩa không dùng ảnh train có nhãn hoặc mask của lớp sản phẩm đích. Tuy nhiên, MuSc là transductive zero-shot vì dùng toàn bộ test set chưa gán nhãn làm nguồn tham chiếu trong quá trình inference.

### 5.2. Training-free khác zero-shot như thế nào?

Zero-shot nói về việc không huấn luyện trên lớp hoặc dataset đích. Training-free nhấn mạnh rằng MuSc không có thêm giai đoạn học anomaly-specific. Backbone CLIP vẫn là mô hình đã được pretrain trước đó.

### 5.3. Dùng test set để so khớp có phải data leakage không?

MuSc dùng ảnh test nhưng không dùng label hoặc mask khi inference. Đây là protocol transductive được phương pháp công bố sử dụng. Ground truth chỉ được dùng sau inference để tính metric.

### 5.4. Tại sao backbone tên ViT-L/14@336 nhưng ảnh đầu vào là 518 x 518?

Tên backbone phản ánh cấu hình pretraining của CLIP. MuSc chính thức resize ảnh thành 518 x 518 để tạo lưới patch dày hơn cho anomaly localization. Đồ án giữ đúng cấu hình repository để tái lập công bằng.

### 5.5. Tại sao chỉ lấy 30% khoảng cách nhỏ nhất trong MSM?

Một patch normal chỉ cần tìm được một nhóm ảnh đủ tương tự, không cần giống toàn bộ test set. Điều này giúp giảm ảnh hưởng của biến thiên hình dạng normal giữa các ảnh.

### 5.6. RsCIN có thay đổi heatmap không?

Không. RsCIN chỉ hiệu chỉnh score cấp ảnh. Anomaly map và các metric cấp pixel được hình thành trước RsCIN.

### 5.7. Vì sao kết quả cấp pixel khớp hơn cấp ảnh?

Kết quả cấp pixel được tính trực tiếp từ anomaly map và có sai lệch mean nhỏ so với baseline repository. Cấp ảnh còn đi qua maximum pooling và RsCIN. Sai lệch tập trung ở đây cho thấy RsCIN hoặc khác biệt môi trường là giả thuyết hợp lý, nhưng đồ án không tuyên bố đã chứng minh nguyên nhân.

### 5.8. Vì sao không chạy WinCLIP làm baseline khác?

Mục tiêu của đồ án là tái lập có kiểm soát một bài báo CV. Kết quả đã được đối chiếu với cả bài báo và repository trên hai benchmark. WinCLIP là một phương pháp hoàn chỉnh dùng text prompt, không phải backbone có thể thay trực tiếp vào MuSc. Chạy thêm WinCLIP cần một protocol riêng và không phải yêu cầu bắt buộc của đồ án.

### 5.9. Vì sao PatchCore không phải baseline so sánh trực tiếp phù hợp?

PatchCore cần tập ảnh normal của lớp đích để xây dựng memory bank, trong khi MuSc dùng test set chưa gán nhãn. Hai phương pháp dùng nguồn dữ liệu và giả định khác nhau, nên so sánh trực tiếp cần thiết kế protocol riêng.

### 5.10. Tại sao chỉ chạy top 1% pooling trên MVTec AD?

Đây là focused extension sau khi phần tái lập bắt buộc trên cả MVTec AD và VisA đã hoàn thành. MVTec AD được chọn làm benchmark chính cho controlled comparison. Kết quả trên toàn bộ 15 lớp không ủng hộ giả thuyết, nên không mở rộng thêm sang VisA chỉ để tìm kết quả thuận lợi.

### 5.11. Kết quả âm của top 1% có giá trị gì?

Kết quả âm cho thấy trực giác giảm ảnh hưởng của một điểm nóng không đủ để đảm bảo xếp hạng ảnh tốt hơn. Phân tích định tính còn chỉ ra hai cơ chế đối lập: top 1% có thể giảm đỉnh cục bộ nhưng vẫn giữ đáp ứng rộng trên toàn vật thể.

### 5.12. F1-max có dùng được trực tiếp khi triển khai không?

Không hoàn toàn. F1-max dùng threshold tối ưu hậu nghiệm trên benchmark, nên thể hiện khả năng phân tách tốt nhất có thể. Khi triển khai cần chọn threshold bằng dữ liệu validation hoặc một quy tắc đã khóa trước.

### 5.13. Vì sao AUROC cao nhưng heatmap vẫn xấu?

Pixel AUROC đánh giá thứ tự score giữa pixel lỗi và pixel normal trên mọi threshold. Một heatmap có thể xếp pixel lỗi cao hơn phần lớn nền nhưng vẫn kích hoạt lan rộng quanh vật thể, nên biên định vị không nhất thiết chặt.

### 5.14. Những thay đổi code có làm thay đổi thuật toán không?

Hai thay đổi chính là sửa xung đột import với package `datasets` và giải phóng tensor dư để giảm CPU RAM. Công thức tính điểm không được thay đổi. Tuy nhiên, chưa có A/B test matched-environment cho memory patch, nên báo cáo vẫn công bố đây là một giới hạn reproducibility.

### 5.15. Đóng góp chính của đồ án là gì?

Đây không phải đóng góp thuật toán mới. Đóng góp của đồ án là một quy trình tái lập có thể audit trên hai benchmark, phân tích định lượng và định tính các failure case, kiểm tra riêng RsCIN, và một controlled extension được báo cáo trung thực dù kết quả không ủng hộ giả thuyết.

### 5.16. Có thể upload một ảnh mới hoặc một category mới không?

Một ảnh cô lập không phải input đúng của MuSc vì mutual scoring cần các ảnh khác cùng category làm nguồn tham chiếu. Một category mới có thể được xử lý theo zero-shot protocol nếu cung cấp một tập nhiều ảnh chưa gán nhãn của category đó. Các category khác nhau cần được xử lý riêng. Nếu hệ thống dùng một reference pool cố định để nhận từng ảnh mới, đó là một adaptation triển khai và không còn đúng hoàn toàn với protocol tái lập trong đồ án.

### 5.17. Có thể dùng MuSc cho ảnh X-quang hoặc ảnh nha khoa không?

Về kỹ thuật có thể chạy thử nếu có một tập nhiều ảnh cùng loại chụp, cùng vùng giải phẫu và điều kiện thu nhận đủ tương đồng. Tuy nhiên, đồ án và bài báo chỉ đánh giá trên dữ liệu kiểm định công nghiệp, nên không có bằng chứng để khẳng định hiệu quả trên nha khoa. Ảnh y khoa có domain gap lớn so với ảnh tự nhiên dùng để pretrain CLIP; biến thiên giải phẫu, máy chụp, góc chụp, miếng trám hoặc implant cũng có thể bị xem nhầm là anomaly. Muốn sử dụng nghiêm túc cần một dental dataset có ground truth, đánh giá lại image-level và pixel-level metrics, và kiểm định lâm sàng. MuSc chỉ tạo anomaly score và vùng nghi ngờ, không phải mô hình chẩn đoán.

### 5.18. Đồ án có reproduce bài MuSc-V2 ở arXiv:2511.10047 không?

Không. Đồ án hiện tại reproduce MuSc bản ICLR 2024, thể hiện qua các thành phần LNAMD, MSM và RsCIN cùng thí nghiệm 2D trên MVTec AD và VisA. MuSc-V2 là công trình năm 2025 mở rộng sang dữ liệu 2D/3D và multimodal, với các thành phần khác như IPG, SNAMD, CAE và RsCon. Không dùng kết quả của đồ án hiện tại để tuyên bố đã reproduce MuSc-V2.

## 6. Tóm tắt 60 giây

"Đồ án tái hiện phương pháp MuSc bản ICLR 2024 cho phát hiện và phân vùng bất thường zero-shot trên ảnh công nghiệp. Input của MuSc là một tập nhiều ảnh chưa gán nhãn thuộc cùng một category, không phải một ảnh đơn lẻ; output cho từng ảnh gồm anomaly score và anomaly map. MuSc dùng CLIP để trích xuất đặc trưng, LNAMD để kết hợp nhiều mức ngữ cảnh, MSM để tìm patch hiếm giữa các ảnh test và RsCIN để hiệu chỉnh score cấp ảnh. Em đã chạy toàn bộ 15 category MVTec AD và 12 category VisA. Sai lệch pixel mean lớn nhất so với baseline repository là 0,0044 điểm phần trăm; sai lệch cấp ảnh lớn hơn, nhưng nguyên nhân chưa được xác định bằng thí nghiệm kiểm soát. Hình định tính cho thấy heatmap có thể bám vùng lỗi nhỏ hoặc lan sang cấu trúc lân cận. Thử nghiệm Top 1% pooling cho kết quả tổng hợp thấp hơn maximum pooling trên MVTec AD. Vì vậy, metric cần được đọc cùng protocol, failure cases và điều kiện reproducibility."

## 7. Checklist trước khi trình bày

- Xác nhận thời lượng được giao và chọn phiên bản đầy đủ hoặc rút gọn.
- Mở thử PDF trên máy sẽ dùng để trình chiếu.
- Kiểm tra font, hình ảnh và hyperlink repository.
- Đăng nhập sẵn vào tài khoản có quyền xem private repository nếu cần demo.
- Chuẩn bị câu trả lời về zero-shot, transductive inference, RsCIN và lý do không chạy WinCLIP.
- Nói rõ ngay từ slide 3 rằng input là một tập nhiều ảnh cùng category, không phải một ảnh đơn lẻ.
- Nếu được hỏi về domain y khoa, chỉ nói đây là khả năng cần đánh giá lại, không khẳng định có thể dùng lâm sàng.
- Gọi đúng công trình chính là MuSc, ICLR 2024; không nhầm với MuSc-V2, arXiv 2025.
- Không mở notebook hoặc chạy lại Colab trong lúc trình bày trừ khi giảng viên yêu cầu.
- Kết thúc bằng ba thông điệp chính thay vì chỉ nói "em xin hết".
