# Ghi chú kiến thức thuyết trình MuSc - CS2203

## Phạm vi và nguồn đối chiếu

Ghi chú phục vụ việc hiểu và bảo vệ nội dung của đủ **17 slide MuSc ICLR 2024**. Nội dung gồm kiến thức nền, giải thích từng slide, cách đọc kết quả, những điểm cần nói chính xác và câu hỏi kiểm tra mức độ hiểu. Đây là tài liệu ôn kiến thức, bổ sung cho dàn ý nói, không thay thế báo cáo thực nghiệm.

Nguồn được đối chiếu ngày 02/11/2026:

- **[S]** [PDF 17 slide](2026-10-17-musc-reproduction-slides-vi.pdf), bản đã cập nhật hình ngày 02/11; [LaTeX nguồn](2026-10-15-musc-reproduction-slides-vi.tex).
- **[O]** [Outline thuyết trình](2026-10-18-musc-presentation-outline-vi.md).
- **[P]** Xurui Li, Ziming Huang, Feng Xue và Yu Zhou. *MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images*. ICLR 2024. [PDF proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/096b1019463f34eb241e87cfce8dfe16-Paper-Conference.pdf). Đã đọc phần chính và phụ lục; mục và số bảng bên dưới theo bản này.
- **[C]** [Mã nguồn MuSc tại commit b76b93da8bd3096a99964a96ae29d46f197a0651](https://github.com/xrli-U/MuSc/tree/b76b93da8bd3096a99964a96ae29d46f197a0651), đối chiếu trực tiếp LNAMD, MSM, RsCIN, pipeline và metric.
- **[E01]** [MVTec AD định lượng](../2026-10-12-musc-e01-result.md); **[E01Q]** [MVTec AD định tính](../2026-10-12-musc-e01q-result.md).
- **[E02]** [VisA định lượng](../2026-10-14-musc-e02-result.md); **[E02Q]** [VisA định tính và RsCIN](../2026-10-15-musc-e02q-result.md).
- **[E04]** [Thử nghiệm top 1%](../2026-10-17-musc-e04-result.md); **[E04Q]** [Phân tích hình của top 1%](../2026-10-17-musc-e04q-result.md).
- **[Q-fix]** [Đính chính hình và diễn giải ngày 02/11](../2026-11-02-musc-qualitative-figure-correction.md), được ưu tiên khi mâu thuẫn với mô tả định tính cũ.

Các ví dụ số ghi là **minh họa** được tạo để giải thích phép tính, không phải kết quả MuSc. Những nội dung mang tính giải thích cơ chế được phân biệt với nguyên nhân đã được kiểm chứng. Không chạy lại inference MuSc trong lần chuẩn bị ghi chú này.

## 1. Những điểm cần làm rõ trong outline

Outline đã khớp thứ tự và câu chuyện của 17 slide. Phần cần bổ sung chủ yếu là kiến thức phía sau câu nói và giới hạn suy luận.

| Vị trí | Điểm dễ hiểu nhầm | Cách hiểu và diễn đạt chính xác |
|---|---|---|
| Slide 2-3 | “Anomaly hiếm” nghĩa là đa số ảnh test phải normal | Cốt lõi là **pattern lỗi khó lặp lại giữa nhiều ảnh**, trong khi vùng normal vẫn phong phú, kể cả bên trong ảnh lỗi. Paper ghi tỷ lệ pixel normal là 97,26% trên MVTec AD và 99,45% trên VisA. Không đặt điều kiện đa số ảnh phải normal. [P, §1] |
| Slide 4 | MuSc “không dùng ảnh normal đích” | MuSc không cần **tập normal train được xác nhận riêng**. Tập test chưa gán nhãn vẫn chứa ảnh và patch normal được dùng làm tham chiếu. |
| Slide 4 | PatchCore xây memory bank tức là fine-tune backbone | Memory bank là lưu/chọn feature normal, không đồng nghĩa với gradient training. Cột “Học anomaly-specific” đang gộp cả xây mô hình tham chiếu; cần giải thích khi được hỏi. |
| Slide 5, 9 | Layer 5, 11, 17, 23 là layer thứ 5, 11, 17, 23 | Đây là index từ 0 trong config; tương ứng đầu ra transformer block thứ **6, 12, 18, 24**. [C, `models/musc.py`] |
| Slide 6 | LNAMD là thay kích thước patch input hoặc giới hạn attention | Patch input vẫn 14 × 14. LNAMD tổng hợp neighborhood trên **token đã trích xuất**, không sửa self-attention hay huấn luyện thêm. |
| Slide 7 | 30% là lấy 30% patch hoặc lọc trước 30% ảnh giống nhau nhất | Với **từng query patch**, mỗi ảnh khác đóng góp một nearest-patch distance; chọn 30% nhỏ nhất trong vector khoảng cách đó. Nhóm ảnh được chọn có thể khác nhau giữa các query patch. |
| Slide 10-11 | “Baseline exact” và chênh lệch rất nhỏ chứng minh tái lập bitwise | “Exact” chỉ nên hiểu là **mốc tham chiếu đã ghim**. Bảng repo làm tròn hai chữ số; môi trường không trùng, không có raw official outputs để chứng minh khớp bitwise. |
| Slide 10-11 | Đã chứng minh paper và README chỉ khác ở một lần chạy | Hai nguồn cùng mô tả cấu hình chính nhưng không có lịch sử từng run để xác nhận đầy đủ nguyên nhân chênh lệch. Câu an toàn: “Paper là mốc công bố; đồ án audit theo bảng repo tại commit đã ghim.” |
| Slide 12-13 | AUROC 0,9999 là Image AUROC hoặc accuracy | Đây là **Pixel AUROC tính trong một ảnh có lỗi**. Hai ca MVTec dùng heatmap 8-bit đã lưu; hai ca VisA dùng anomaly map gốc. Không thay thế metric category. |
| Slide 14 | RsCIN làm hại `pcb3` nên đã giải thích được sai lệch với repo | Đây là hai phép so sánh khác: trước/sau RsCIN trong cùng run và run tái lập/official reference. Phép thứ nhất không chứng minh nguyên nhân của phép thứ hai. |
| Slide 15 | Top 1% làm heatmap bớt nóng hoặc cải thiện localization | Cùng anomaly map; chỉ thay image score. Hình minh họa cấu trúc đáp ứng, không phải heatmap trước/sau thay thuật toán. |
| Slide 15 | Score giảm thì không thể tạo false positive mới | Thứ hạng và **threshold F1-max riêng của mỗi biến thể** có thể đổi. FP/TN trên slide là quyết định hậu nghiệm trên benchmark. |
| Slide 16 | “Hoàn thành 27 category” nghĩa là 27 loại defect | Là 27 **product category**: 15 MVTec AD + 12 VisA, mỗi category có thể có nhiều kiểu lỗi. |

**Điểm cần giữ:** outline không che kết quả âm, tách image/pixel, không nhận nguyên nhân khi mới có giả thuyết, và giới hạn kết luận RsCIN ở bốn category đã phân tích. Các nguyên tắc này phù hợp với bằng chứng.

## 2. Kiến thức nền dùng xuyên suốt bài

### 2.1. Anomaly detection, classification và segmentation

Trong đồ án, anomaly là bất thường ngoại quan so với cấu trúc normal của **cùng loại sản phẩm**: vết xước, keo dính, thiếu chân linh kiện, biến dạng hoặc sai cấu trúc. “Khác biệt” trong feature space chỉ là bằng chứng mô hình sử dụng; không phải mọi khác biệt đều là lỗi thực tế. Ánh sáng, góc chụp hoặc dấu nền cũng có thể gây đáp ứng sai.

- **Anomaly classification (AC):** ảnh có bất thường hay không. Output trực tiếp là một số thực, không phải tên kiểu lỗi.
- **Anomaly segmentation (AS):** vị trí nào có dấu hiệu bất thường. Output trực tiếp là một bản đồ score liên tục.
- **Binary prediction:** áp threshold lên score/map để tạo nhãn hoặc mask dự đoán.

Với score cấp ảnh $s_i$, quyết định có thể viết $\hat y_i=\mathbf{1}[s_i>\tau]$. Với map $A_i(u,v)$, mask dự đoán là $\hat M_i(u,v)=\mathbf{1}[A_i(u,v)>\tau_p]$. Threshold cấp ảnh và cấp pixel là hai lựa chọn riêng.

**Score không phải xác suất được hiệu chuẩn.** Giá trị 0,8 không có nghĩa “80% khả năng lỗi”. Một score cao phản ánh bằng chứng bất thường tương đối trong protocol đang dùng.

### 2.2. Ground truth và dữ liệu chưa gán nhãn

Ground truth gồm nhãn ảnh normal/anomaly và mask chỉ vùng lỗi. “Chưa gán nhãn” trong inference nghĩa là thuật toán không sử dụng các nhãn này; benchmark vẫn lưu chúng để đánh giá sau đó.

Ảnh anomaly thường chỉ lỗi ở một vùng nhỏ. Phần lớn patch còn lại vẫn normal và có thể giúp chấm điểm ảnh khác. Đây là lý do MuSc dùng được tập test pha trộn normal/anomaly mà không cần xác định trước ảnh nào sạch. [P, §1, §3.2]

### 2.3. Zero-shot, training-free, frozen backbone và transductive

| Thuật ngữ | Nghĩa trong đồ án |
|---|---|
| Pretraining | CLIP đã học biểu diễn từ dữ liệu image-text trước khi dùng cho MuSc |
| Frozen backbone | Trọng số CLIP giữ nguyên khi inference MuSc |
| Fine-tuning | Cập nhật trọng số từ một nhiệm vụ/dataset tiếp theo; baseline MuSc không có bước này |
| Zero-shot | Không cần mẫu huấn luyện của category đích theo setting đang xét; không có nghĩa mô hình chưa từng học dữ liệu |
| Training-free | MuSc không thêm giai đoạn huấn luyện anomaly-specific; vẫn thừa hưởng backbone đã pretrain |
| Prompt-free | Không cần text prompt khi chạy MuSc; không phủ nhận vai trò ngôn ngữ trong pretraining CLIP |
| Transductive | Kết quả của một ảnh phụ thuộc các ảnh test chưa gán nhãn khác trong tập đang xử lý |
| Inductive | Quy tắc đã xây dựng có thể áp dụng cho một ảnh mới độc lập với các ảnh test còn lại |

Dùng ảnh test chưa gán nhãn trong transductive inference phù hợp protocol công bố. Dùng nhãn/mask test để chọn feature, tuning hoặc huấn luyện là một việc khác và có thể làm sai protocol. F1-max có dùng nhãn trong **đánh giá**, vì vậy phải tách threshold hậu nghiệm khỏi cơ chế tạo score.

### 2.4. Feature, patch token và class token

Feature là vector biểu diễn nội dung ảnh. Khoảng cách giữa feature hữu ích khi biểu diễn đã học làm các cấu trúc tương tự nằm gần nhau. MuSc không so sánh RGB từng pixel trực tiếp.

Vision Transformer chia ảnh thành patch, biến patch thành token, bổ sung thông tin vị trí rồi xử lý bằng các transformer block. Self-attention cho phép token kết hợp thông tin từ các vị trí khác.

- **Patch token:** gắn với một vị trí trên lưới, dùng cho localization.
- **Class token:** token tổng hợp biểu diễn cấp ảnh; trong MuSc dùng feature class token đã chiếu tuyến tính ở layer cuối cho RsCIN.
- Patch token sau transformer đã có ngữ cảnh; không nên xem nó là mô tả hoàn toàn độc lập của một ô ảnh thô.
- Class token không phải nhãn category hay nhãn normal/anomaly. Nó vẫn là vector.

CLIP học liên kết ảnh và văn bản bằng contrastive pretraining. MuSc khai thác **image encoder** của CLIP; không cần nhập câu “normal bottle” hay “damaged bottle”. Kiến thức nền này dựa trên Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021, [PMLR](https://proceedings.mlr.press/v139/radford21a.html).

### 2.5. Cấu hình ViT-L/14@336 và input 518

`L` là Large; `14` là cạnh patch input; `336` là cấu hình độ phân giải của biến thể pretrained. MuSc chính thức chạy ảnh **518 × 518**, nên tên backbone và kích thước inference không mâu thuẫn. [P, §4]

$$
518/14=37,\qquad M=37\times37=1369.
$$

Mỗi ảnh có lưới 1.369 patch token ở mỗi stage, cùng class token. Bốn stage lấy sau block 6, 12, 18, 24. Chúng có cùng lưới không gian trong backbone này nhưng khác mức biểu diễn; không phải bốn ảnh đầu vào có kích thước khác nhau.

LNAMD dùng ba degree; kết hợp với bốn stage tạo **12 nhánh chấm điểm**. Điều này không có nghĩa chạy 12 backbone độc lập hay huấn luyện 12 mô hình.

### 2.6. Chuẩn hóa và khoảng cách

Code L2-normalize aggregated patch feature trước MSM và normalize image feature trước xây similarity RsCIN. Với hai vector đơn vị $u,v$:

$$
\|u-v\|_2^2=2-2u^\top v.
$$

Vì vậy Euclidean distance nhỏ tương ứng cosine similarity lớn trên các vector đã normalize. Hai chỗ dùng độ tương tự theo hai hướng đọc khác nhau: **MSM distance lớn là lạ hơn; RsCIN similarity lớn là gần nhau hơn**. Không được suy ra hai metric cho cùng score sau khi lấy mean, vì phép căn là phi tuyến.

## 3. Kiến thức theo đủ 17 slide

### Slide 1 - Trang tiêu đề

**Ý chính:** đồ án tái hiện và phân tích một phương pháp đã công bố, không nhận LNAMD/MSM/RsCIN là thuật toán mới của đồ án.

Tên MuSc gắn với **Mutual Scoring**, nghĩa là ảnh chưa gán nhãn chấm điểm lẫn nhau. Công trình chính là MuSc ICLR 2024 của Li, Huang, Xue và Zhou. Không nhầm với các công trình mở rộng có tên gần giống.

**Phạm vi đóng góp của đồ án:** tái lập hai benchmark, giữ bằng chứng chạy, đối chiếu số liệu, phân tích thành công/thất bại, xem ảnh hưởng RsCIN và thử top 1% có kiểm soát.

**Tự kiểm tra:** phân biệt được “đóng góp paper gốc” với “phần việc của đồ án” trong khoảng 20 giây.

### Slide 2 - Bối cảnh kiểm định công nghiệp

**Kiến thức cần hiểu:** supervised learning thường cần ví dụ đại diện cho lớp cần nhận biết. Trong kiểm định, loại lỗi có thể hiếm và chưa biết trước, nên khó thu đủ ảnh lỗi; mask pixel còn tốn công hơn nhãn ảnh. MuSc khai thác biểu diễn pretrained và tính lặp lại của normal để giảm yêu cầu dữ liệu đích.

Bốn ô hình có vai trò khác nhau:

1. Ảnh gốc là dữ liệu quan sát.
2. Ground truth là đáp án người gán nhãn, chỉ dùng đánh giá.
3. Heatmap là cách tô màu anomaly map.
4. Overlay đặt heatmap lên ảnh gốc để nhìn vị trí tương quan.

Màu nóng có nghĩa score tương đối cao theo phép chuẩn hóa hiển thị, không tự động có nghĩa “ảnh đã bị kết luận lỗi”. Hình trên slide là `leather/glue/011`: vết keo nhỏ trên nền da có pattern lặp lại, phù hợp để giải thích trực giác mutual scoring.

**Giới hạn cần hiểu:** lỗi hiếm trong sản xuất không đồng nghĩa benchmark phải có tỷ lệ ảnh lỗi thấp. Dữ liệu benchmark thường chủ động chứa nhiều ảnh lỗi để đánh giá.

**Tự kiểm tra:** giải thích được vì sao ảnh lỗi vẫn có thể cung cấp nhiều patch normal.

### Slide 3 - Bài toán và đầu ra

Input $D_c=\{x_1,\ldots,x_N\}$ là nhiều ảnh cùng category $c$. Với mỗi ảnh, output gồm một image score và một pixel map. Category ở đây là `bottle`, `leather`, `pcb4`..., không phải `good`, `scratch` hay `glue`.

**Tại sao cùng category?** Khoảng cách chỉ có nghĩa khi so sánh những cấu trúc đáng lẽ có normal pattern tương thích. Trộn chai, vít và PCB làm thay đổi nguồn tham chiếu và điều kiện đánh giá; baseline xử lý chúng riêng.

**Tại sao không một ảnh?** MSM loại chính ảnh query, rồi dùng các ảnh còn lại để chấm điểm. Với một ảnh, không còn nguồn tham chiếu. Với một tập rất nhỏ, code còn có ràng buộc số neighbor và số ảnh trong interval; “nhiều ảnh” là yêu cầu phương pháp, không chỉ yêu cầu giao diện demo.

**Giả định chính xác:** normal pattern có đủ đối sánh giữa nhiều ảnh; anomaly pattern không có quá nhiều đối sánh gần. Nếu một vết lỗi giống nhau lặp lại nhiều, nó có thể bị chấm thấp. Nếu normal appearance hiếm, nó có thể bị chấm cao.

**Tự kiểm tra:** dự đoán cho một ảnh có thể thay đổi khi thêm ảnh khác vào tập dù ảnh đó không đổi. Giải thích được nguyên nhân bằng MSM và RsCIN.

### Slide 4 - Các protocol anomaly detection

| Phương pháp | Cách suy luận cơ bản | Kiến thức cần nắm để so sánh |
|---|---|---|
| PatchCore | So patch test với memory bank feature normal | Cần ảnh normal đích; không cần ví dụ của mọi loại lỗi; memory bank không đồng nghĩa fine-tuning |
| WinCLIP zero-shot | Dùng prompt ensemble và feature image/window/patch liên kết với text | Không train theo category đích; vẫn cần ngôn ngữ mô tả trạng thái normal/abnormal |
| AnomalyCLIP | Học object-agnostic prompt trên dữ liệu phụ trợ | Zero-shot với dataset đích nhưng có học anomaly-specific trên dữ liệu khác |
| MuSc | So khớp patch giữa ảnh test chưa gán nhãn, sau đó re-score cấp ảnh | Không cần normal train riêng hoặc text prompt; phụ thuộc test population |

**Điểm cần tránh:** training-free không tự động nghĩa là không dùng dữ liệu đích; MuSc sử dụng ảnh test đích. Cũng không thể nhìn riêng nhãn “zero-shot” để kết luận mọi phương pháp có cùng nguồn thông tin.

PatchCore và WinCLIP là phương pháp hoàn chỉnh; CLIP là backbone/biểu diễn nền. “Thay backbone bằng WinCLIP” không phải một thay thế tương đương.

Nguồn: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022, [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html); Jeong et al., *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation*, CVPR 2023, [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html); Zhou et al., *AnomalyCLIP*, ICLR 2024, [bản tác giả](https://arxiv.org/abs/2310.18961).

**Tự kiểm tra:** gọi đúng nguồn tham chiếu của cả bốn phương pháp mà không biến bảng protocol thành bảng xếp hạng.

### Slide 5 - Pipeline MuSc

Luồng xử lý cần giải thích được từ đầu đến cuối:

1. Image encoder trích patch feature ở bốn stage và image feature cuối.
2. LNAMD tạo biểu diễn neighborhood với $r=1,3,5$ cho mỗi stage.
3. MSM chấm từng patch bằng các ảnh khác trong cùng category.
4. Lấy mean score qua bốn stage và ba degree, thu một score cho mỗi vị trí patch.
5. Reshape thành lưới 37 × 37 rồi nội suy thành map 518 × 518.
6. Lấy maximum để có image score ban đầu.
7. RsCIN dùng similarity của image feature để hiệu chỉnh image score.

**Hai nhánh khác nhau:** patch token đi qua LNAMD/MSM để tạo map; class token đi vào RsCIN. RsCIN nhận cả score ban đầu và image similarity, không tự tạo map.

**Chi tiết paper/code:** paper viết $c_i=\max_m a_i^m$ ở lưới patch. Code ghim lấy max trên **anomaly map đã nội suy**. Đây là hai cách diễn đạt cùng vai trò tổng hợp, nhưng không nên khẳng định luôn bằng nhau từng giá trị: bilinear interpolation có thể thay đổi cực đại lấy mẫu. E04 đối chiếu theo cách làm của code, từ cùng map đã nội suy. [P, Eq. 3; C, `models/musc.py`]

**Tự kiểm tra:** chỉ ra ngay đầu ra nào còn tồn tại nếu bỏ RsCIN: map giữ nguyên, image score còn maximum chưa hiệu chỉnh.

### Slide 6 - LNAMD

**Tên đầy đủ:** Local Neighborhood Aggregation with Multiple Degrees.

Với từng vị trí trên lưới token, tổng hợp feature trong neighborhood $r\times r$ bằng average pooling. Code dùng local patch extraction và adaptive average pooling; sau đó feature được L2-normalize trước chấm điểm. Không cập nhật trọng số pretrained. [P, §3.1; C, `models/modules/_LNAMD.py`]

| Degree | Lân cận trực tiếp trên lưới token | Trực giác | Đánh đổi |
|---|---|---|---|
| $r=1$ | Một vị trí | Giữ tín hiệu nhỏ, cục bộ | Thiếu ngữ cảnh cho cấu trúc lớn |
| $r=3$ | 3 × 3 vị trí | Kết hợp dấu hiệu với vùng gần | Có thể làm loãng lỗi rất nhỏ |
| $r=5$ | 5 × 5 vị trí | Thêm cấu trúc rộng | Dễ làm mờ tương phản lỗi nhỏ |

Nếu chỉ quy đổi theo footprint patch input, 3 × 3 và 5 × 5 tương ứng 42 × 42 và 70 × 70 pixel. Tuy nhiên **đó không phải receptive field thực tế đầy đủ** vì token đã qua self-attention.

**Vì sao giữ cả ba?** Lỗi có nhiều kích thước. Paper chọn cân bằng classification/segmentation trên hai dataset, không chứng minh $\{1,3,5\}$ tối ưu mọi metric. Ví dụ Table 3 có cấu hình $r=5$ đạt Pixel AUROC MVTec cao hơn tổ hợp ba degree, nhưng classification thấp hơn. [P, Table 3, Appendix A.2.2]

**Không làm gì:** LNAMD không biết patch nào lỗi, không dùng mask, không tự chọn degree theo kích thước GT. Baseline tính đủ các degree rồi kết hợp **score**.

**Tự kiểm tra:** giải thích vì sao tăng $r$ có thể tốt cho lỗi lớn và xấu cho vết xước nhỏ mà không nói “$r$ lớn luôn tốt hơn”.

### Slide 7 - MSM

**Tên đầy đủ:** Mutual Scoring Mechanism. Đây là phần quan trọng nhất cần nắm chắc.

#### Bước 1: một patch được từng ảnh khác chấm điểm

$$
d_{i,l}^{m,r}(j)=\min_n\left\|\hat p_{i,l}^{m,r}-\hat p_{j,l}^{n,r}\right\|_2,\qquad j\ne i.
$$

| Ký hiệu | Nghĩa |
|---|---|
| $i$ | Ảnh đang được đánh giá |
| $j$ | Một ảnh tham chiếu khác trong cùng category |
| $m$ | Vị trí query patch trong ảnh $i$ |
| $n$ | Vị trí patch ứng viên trong ảnh $j$ |
| $l$ | Stage trích feature, trong bốn stage |
| $r$ | Degree LNAMD 1, 3 hoặc 5 |
| $\hat p$ | Aggregated patch feature dùng để so khoảng cách |
| $\min_n$ | Chọn patch giống nhất trong **ảnh $j$**, không buộc cùng tọa độ |

Mỗi ảnh khác cho một khoảng cách. Một query patch vì thế nhận vector dài $N-1$. Không so query với chính nó vì đối sánh bản thân sẽ có distance 0 và làm mất ý nghĩa phát hiện.

Tìm mọi vị trí giảm yêu cầu alignment cứng. Nó không tạo bất biến tuyệt đối với xoay/scale: feature của cùng cấu trúc sau xoay vẫn có thể khác đáng kể.

#### Bước 2: Interval Average của 30% nhỏ nhất

Sắp vector khoảng cách tăng dần $d_{(1)}\le\cdots\le d_{(N-1)}$ rồi lấy:

$$
K=\lfloor0.3(N-1)\rfloor,\qquad a_{i,l}^{m,r}=\frac1K\sum_{k=1}^K d_{(k)}.
$$

Cách lấy phần nguyên này theo code ghim. Nó yêu cầu $K>0$; không phải một công thức tự xử lý được mọi tập nhỏ.

**Ví dụ số minh họa:** có 11 ảnh, nên mỗi patch nhận 10 khoảng cách và $K=3$.

- Patch normal: `[0.10, 0.12, 0.14, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]`. Mean thấp nhất 30% là 0,12; mean cả vector là 0,526. Một số ảnh normal khác pose không kéo score lên quá nhiều.
- Patch anomaly: `[0.05, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]`. Có một đối sánh tình cờ gần, nhưng mean thấp nhất 30% vẫn khoảng 0,433. Nếu dùng global minimum thì score chỉ 0,05 và dễ bỏ sót.

Hai mục tiêu được cân bằng: không để **một** đối sánh gần quyết định tất cả; cũng không bắt normal patch giống **mọi** ảnh. Không có bảo đảm anomaly luôn điểm cao: nếu đủ nhiều patch lỗi lặp lại, ngay nhóm 30% cũng có thể rất gần.

**30% không phải:** tỷ lệ ảnh anomaly, tỷ lệ diện tích lỗi, confidence threshold hay top pixel pooling. Đây là hyperparameter chọn khoảng distance, được paper kiểm tra trong Figure 7 và kế thừa nguyên cấu hình cho đồ án.

#### Bước 3: kết hợp 12 nhánh

$$
a_i^m=\frac1{4\cdot3}\sum_{l=1}^{4}\sum_{r\in\{1,3,5\}}a_{i,l}^{m,r}.
$$

Mean của **score** qua stage/degree tạo score patch cuối. Không phải lấy mean raw token của tất cả stage trước rồi chỉ chạy MSM một lần.

**Vì sao tốn tài nguyên?** Với $N$ ảnh, $M$ patch/ảnh và feature dimension $C$, phép so sánh trực tiếp có khối lượng cỡ $O(12N(N-1)M^2C)$. Đây là phân tích bậc độ lớn, không phải thời gian đo. Code xử lý theo query/nhánh để giảm bộ nhớ đỉnh, nên không cần giữ toàn bộ tensor khoảng cách toàn dataset đồng thời.

**Tự kiểm tra:** giải thích bằng lời khác nhau giữa $\min_n$, chọn 30% trên $j$, và mean trên $l,r$; đây là ba trục tổng hợp khác nhau.

### Slide 8 - RsCIN

**Tên đầy đủ:** Re-scoring with Constrained Image-level Neighborhood.

Maximum nhạy với một đáp ứng cực đại trên ảnh normal. Trong khi đó, image feature toàn cục có thể ít nhạy hơn với nhiễu rất cục bộ. RsCIN dùng quan hệ giữa ảnh để điều chỉnh score, dựa trên giả định các ảnh có global feature gần nhau nên có mức bất thường tương tự. [P, §3.3]

Với image feature đơn vị $f_i$, $W_{ij}=f_i^\top f_j$ là cosine similarity. Với mỗi window size $k$, giữ $k$ ảnh tương tự nhất và chuẩn hóa trọng số theo hàng:

$$
q_i^{(k)}=\frac{\sum_j M_k(i,j)W_{ij}c_j}{\sum_j M_k(i,j)W_{ij}},
\qquad \hat c_i=\frac{c_i+q_i^{(k_1)}+q_i^{(k_2)}}3.
$$

$M_k$ là **mask của graph ảnh**, không phải ground-truth defect mask. “Constrained” nghĩa là chỉ một neighborhood nhỏ tác động, tránh trộn score từ quá nhiều ảnh không phù hợp. “Multi-window” nghĩa là nhiều kích thước neighborhood image-level, không phải cửa sổ pixel.

**Paper và code:** paper viết window $\{2,3\}$ cho MVTec AD và $\{8,9\}$ cho VisA, kèm riêng self-score. Code dùng `[1,2,3]` và `[1,8,9]`. Window $k=1$ thường giữ chính ảnh vì self-similarity lớn nhất, nên đóng vai trò self-score. Neighbor list trong code **có tính self**, không được diễn giải thành luôn có 2/3 hay 8/9 ảnh khác ngoài query. Trường hợp similarity hòa cần đọc theo chọn neighbor thực tế. [C, `models/modules/_RsCIN.py`]

Code min-max normalize vector score cấp category trước RsCIN. Đây không phải bước học threshold bằng GT.

**Ví dụ số minh họa:** image score normal nhiễu là 0,9; hai score tổng hợp neighborhood là 0,3 và 0,2. Kết quả $(0,9+0,3+0,2)/3\approx0,467$ có thể giảm false positive. Nhưng nếu một ảnh lỗi nhỏ có global appearance giống normal, cùng cơ chế có thể kéo score lỗi xuống. Đó là đánh đổi, không phải quy tắc chỉ giảm noise.

**Điểm cần nói chắc:** RsCIN không làm thay đổi map, không fine-tune CLIP và không đảm bảo luôn tốt. Tăng $k$ không chắc tốt hơn vì có thể làm mượt quá mức và trộn trạng thái lỗi khác nhau. [P, Table 5, Figure 8]

**Tự kiểm tra:** phân biệt neighbor của MSM và RsCIN: MSM chọn evidence theo từng patch; RsCIN chọn theo global image feature.

### Slide 9 - Dataset, metric và protocol

#### Dữ liệu và cách chạy

| Nội dung | MVTec AD | VisA |
|---|---|---|
| Toàn dataset | 5.354 ảnh, 15 category | 10.821 ảnh, 12 category |
| Nhóm | 10 object, 5 texture | PCB phức tạp, nhiều vật thể, vật thể gần căn chỉnh |
| Phạm vi quantitative baseline | 15 category, 1.725 ảnh test | 12 category, 2.162 ảnh test |
| Dữ liệu train | Không đưa vào baseline MuSc | Không đưa vào baseline MuSc |
| Ground truth | Nhãn ảnh và mask để tính metric | Nhãn ảnh và mask để tính metric |

VisA có 9.621 ảnh normal và 1.200 ảnh anomaly trong toàn bộ dataset. Không đồng nhất 10.821 ảnh toàn bộ với 2.162 ảnh trong test protocol tái lập. Nguồn: [E01], [E02], Bergmann et al., CVPR 2019, [dataset chính thức](https://www.mvtec.com/research-teaching/datasets/mvtec-ad); Zou et al., ECCV 2022, [VisA chính thức](https://github.com/amazon-science/spot-diff).

Năm texture MVTec AD là `carpet`, `grid`, `leather`, `tile`, `wood`; mười object là `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`. Phân nhóm này mô tả benchmark, không quyết định trước nhóm nào dễ hơn.

**Cấu hình cần nhớ:** CLIP ViT-L/14@336; input 518; block 6/12/18/24; LNAMD 1/3/5; MSM thấp nhất 30%; seed 42; `divide_num=1`; batch size 4; commit `b76b93d`.

**Batch size và tập mutual scoring là hai thứ khác nhau.** Batch size 4 phục vụ trích feature theo nhóm để phù hợp tài nguyên. Feature vẫn được gom để MSM chạy trên toàn test set category khi `divide_num=1`. Không phải mỗi ảnh chỉ có ba ảnh cùng minibatch chấm điểm.

Mỗi category chạy subprocess riêng giúp giải phóng bộ nhớ và resume khi ngắt. Log/config/command/commit/attempt được giữ để audit. Seed cố định giúp tái lập nhưng không bảo đảm bitwise giữa software stack khác nhau.

#### TP, FP, TN, FN

Quy ước positive là anomaly:

| Thực tế / Dự đoán | Dự đoán anomaly | Dự đoán normal |
|---|---|---|
| Thực tế anomaly | TP: phát hiện đúng | FN: bỏ sót lỗi |
| Thực tế normal | FP: báo lỗi nhầm | TN: bỏ qua đúng |

Trong cấp pixel, positive là pixel thuộc mask lỗi. FP có thể là vùng nền bị tô nóng, dù image-level classification vẫn đúng.

#### AUROC

Với mỗi threshold:

$$
\mathrm{TPR}=\frac{TP}{TP+FN},\qquad \mathrm{FPR}=\frac{FP}{FP+TN}.
$$

ROC vẽ TPR theo FPR khi thay threshold; AUROC là diện tích dưới đường đó. Có thể hiểu là xác suất một positive có score cao hơn một negative lấy ngẫu nhiên, với score hòa được tính nửa điểm.

**Image AUROC:** so score ảnh anomaly với ảnh normal. **Pixel AUROC:** so score pixel lỗi với pixel normal. 0,5 là mức xếp hạng ngẫu nhiên; 1 là tách hoàn hảo trên tập đang xét. AUROC 0,98 không phải accuracy 98% và không cung cấp threshold triển khai.

Với lỗi rất nhỏ, nền normal có rất nhiều pixel; một vùng FP trông lớn vẫn có thể chiếm tỷ lệ nhỏ trong tổng negative, nên Pixel AUROC cao vẫn đi cùng localization chưa chặt. Không phải cứ mất cân bằng là AUROC sai; vấn đề là metric đang trả lời một câu hỏi xếp hạng khác với chất lượng biên mask.

#### AP

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad \mathrm{Recall}=\frac{TP}{TP+FN}.
$$

Average Precision tóm tắt precision theo các mức recall; trong code dùng `average_precision_score`, không phải lấy trung bình precision tùy ý hoặc mặc nhiên dùng trapezoid AUPRC. AP làm tác động của false positives dễ thấy hơn khi positive ít. AP phụ thuộc prevalence; không so AP giữa các dataset rồi quy hết chênh lệch cho chất lượng thuật toán.

Ví dụ một mask có 100 pixel lỗi, dự đoán trúng 80 và tô nhầm 720 pixel normal: recall 0,8 nhưng precision 0,1. Heatmap có thể “tìm đúng chỗ” nhưng mask vẫn không gọn. Đây là minh họa công thức, không phải số liệu của một ca trên slide.

#### F1-max

$$
F_1=\frac{2PR}{P+R},\qquad F_{1\text{-max}}=\max_\tau F_1(\tau).
$$

F1-max báo hiệu quả tốt nhất đạt được khi chọn threshold hậu nghiệm trên benchmark. Nó cần GT trong bước đánh giá; không có nghĩa MuSc sử dụng GT để tạo feature/score. Triển khai cần quy tắc threshold được chọn và kiểm định riêng. F1-max cao không cho biết recall tại một FPR bắt buộc của nhà máy.

#### PRO và AUPRO

Với mỗi connected defect region $R$, tại threshold $t$:

$$
\mathrm{PRO}(t)=\frac1{\#\mathcal R}\sum_{R\in\mathcal R}\frac{|R\cap\hat M(t)|}{|R|}.
$$

Mỗi vùng lỗi đóng góp một phần tử mean, nên vùng nhỏ không bị vùng lớn lấn át như khi gộp tất cả pixel positive. Code quét 200 threshold, lấy đường PRO-FPR trong miền FPR dưới 0,3 rồi tính diện tích đã chuẩn hóa. Cột slide gọi “PRO” thực chất là **AUPRO trong miền FPR giới hạn**. [C, `utils/metrics.py`]

PRO không phải IoU. Ở từng threshold, nó đo tỷ lệ bao phủ vùng GT; false positive được kiểm soát qua trục FPR. Metric này bổ sung đánh giá vùng nhưng không tự chứng minh biên segmentation chính xác.

#### Cách tổng hợp

Metric được tính **trong từng category** rồi mean không trọng số qua category. Pixel metric của một category trong code gộp pixel của toàn bộ ảnh category đó; không phải trung bình Pixel AUROC từng ảnh. Mean của 15/12 category cũng không phải một AUROC duy nhất tính gộp toàn dataset.

**Tự kiểm tra:** giải thích vì sao cần đồng thời AUROC, AP, F1-max và PRO; chỉ ra metric nào cần threshold tối ưu hậu nghiệm.

### Slide 10 - Kết quả MVTec AD

**Cách đọc bảng:** Repo là giá trị tham chiếu; E01 là run đồ án; delta = E01 - Repo. Các giá trị biểu diễn theo phần trăm; delta nói bằng **điểm phần trăm**.

| Metric | E01 (%) | Delta với repo (điểm phần trăm) |
|---|---:|---:|
| Pixel AUROC | 97,1144 | +0,0044 |
| Pixel F1-max | 62,1580 | -0,0020 |
| Pixel AP | 62,2617 | +0,0017 |
| PRO | 93,4530 | +0,0030 |
| Image AUROC | 97,7340 | -0,0360 |
| Image F1-max | 97,4057 | +0,0357 |
| Image AP | 99,0528 | -0,0172 |

**Hai kết luận khác nhau:** các mean rất gần repo; nhưng không phải mọi metric/category đều khớp đến cùng mức. “60 metric pixel” là 15 category × 4 metric, đều lệch dưới 0,10 điểm. Với cấp ảnh có 10/45 phép so sánh vượt 0,10; lớn nhất là `capsule` F1-max +0,813780 điểm. [E01]

**Texture - Object +6,3305:** Pixel AP macro của texture là 66,4821%, object là 60,1516%; hiệu là +6,3305 điểm. Đây là hai nhóm category không cân bằng và có loại lỗi khác nhau. Không phải cải tiến do một module mới, cũng không chứng minh mọi texture dễ hơn mọi object. Nguồn bảng: [báo cáo, §6.2](../2026-10-15-musc-final-report-draft-vi.md).

**Giới hạn “khớp”:** repo công bố số làm tròn hai chữ số; delta nhỏ hơn 0,005 có thể nằm trong mức làm tròn. Kết luận hợp lý là tái lập gần bảng tham chiếu, không phải chứng minh tương đương số học tuyệt đối.

**Tự kiểm tra:** phân biệt 97,7340% là mức hiệu năng còn -0,0360 điểm là chênh lệch; không gọi -0,0360 là error rate.

### Slide 11 - Kết quả VisA

| Metric | E02 (%) | Delta với repo (điểm phần trăm) |
|---|---:|---:|
| Pixel AUROC | 98,7136 | +0,0036 |
| Pixel F1-max | 48,9005 | +0,0005 |
| Pixel AP | 45,3831 | +0,0031 |
| PRO | 92,4288 | -0,0012 |
| Image AUROC | 92,4001 | -0,1699 |
| Image F1-max | 88,7978 | -0,2622 |
| Image AP | 93,1772 | -0,1328 |

**Kết luận chính:** hoàn thành 12 category/2.162 test images; pixel mean rất gần repo, image mean thấp hơn khoảng 0,13-0,26 điểm. Không che chênh lệch cấp ảnh bằng kết luận “mọi thứ giống hệt”. Sai lệch theo category lớn nhất là `pcb2` Image F1-max -2,506154 điểm. [E02]

**Audit `fryum`:** repo in Pixel AUROC 99,43 nhưng 12 dòng in ra mean thành 98,8383, khác mean công bố 98,71. Giữ nguyên 11 dòng còn lại, giá trị cần có để ra mean in là khoảng 97,89. E02 đo 97,9309.

Đây là **phát hiện không nhất quán nội bộ của bảng tham chiếu**, không phải bằng chứng xác định raw official value đúng là 97,89. Mean và từng dòng đều được làm tròn. Cần giữ giá trị nguồn cùng chú thích, không thay tham chiếu rồi tuyên bố đã khớp.

**Vì sao image lệch hơn pixel?** Các bước tạo map được tái lập rất gần theo metric; image score còn qua maximum và RsCIN. Sai khác software stack hoặc neighbor selection là giả thuyết hợp lý. Chưa có matched-environment rerun để cô lập nguyên nhân, và pixel metrics gần nhau không chứng minh raw map giống hệt từng pixel.

**Tự kiểm tra:** phân biệt được “metric tái lập cao” với “metric tái lập gần official”; một phương pháp rất yếu cũng có thể được reproduce rất chính xác.

### Slide 12 - Hai ví dụ map phù hợp vùng lỗi

**`leather/glue/011`:** đỉnh nóng nằm trong vùng keo; background texture lặp lại giúp cấu trúc bất thường có tương phản. Pixel AUROC trong ảnh là 0,9999.

**`pcb4/Anomaly/032`:** đỉnh cục bộ tương ứng vùng lỗi tròn ở connector. Pixel AUROC trong ảnh là 0,9997. Phần thân board vẫn có đáp ứng; không nên nói toàn heatmap chỉ có pixel lỗi.

**Cách đọc đúng:** xác định lỗi từ ảnh gốc và mask; tìm vị trí đỉnh trên heatmap; nhìn overlay để kiểm tra đáp ứng có nằm trên cấu trúc lỗi hay không; xem có vùng nóng ngoài mask. Số AUROC là bằng chứng bổ sung, không thay thế quan sát này.

**Lưu ý nguồn số:** Leather tính từ heatmap 8-bit đã lưu, PCB4 từ raw map. Lượng tử hóa có thể tạo score hòa; không dùng chữ số 0,9999 để khẳng định độ chính xác raw-score cao hơn một ca khác.

**Mask VisA:** giá trị dương là ID loại lỗi, có thể từ 1 đến 7; hiển thị raw mask trên thang 0-255 làm vùng lỗi gần đen. Hình mới dùng `mask > 0`. Đây là sửa hiển thị; không phải sửa kết quả inference. [Q-fix]

**Tự kiểm tra:** từ “đỉnh nằm trong mask” không suy ra mọi pixel mask được phát hiện tốt hoặc mask dự đoán có IoU cao.

### Slide 13 - Hai failure cases

**`transistor/cut_lead/008`:** đỉnh nóng nằm ở chân bị cắt, nhưng GT còn gồm vùng chân bị thiếu, phần lớn trông giống nền và không được nhấn mạnh. Pixel AUROC trong ảnh chỉ 0,5957. Có một đỉnh đúng không đủ để xếp tốt tất cả pixel lỗi so với pixel normal.

**`fryum/Anomaly/010`:** lỗi mảnh ở biên có đáp ứng yếu hơn vùng nóng mạnh nhất ở nền bên trái. Peak ngoài mask và cách lỗi hơn 100 pixel trên ảnh 518 × 518 theo phép kiểm tra hình đã lưu. Pixel AUROC trong ảnh 0,9250 là thấp nhất trong 100 ảnh lỗi `fryum` của run này, không phải thấp nhất toàn VisA. [Q-fix, E02Q]

**Cơ chế có thể liên quan:** lưới patch hữu hạn, pooling làm loãng lỗi nhỏ, feature bị nhiễu bởi nền/pose, hoặc tham chiếu không phù hợp. Đây là các giải thích khả dĩ, không phải các nguyên nhân đã được tách bằng ablation trong đồ án.

**Điểm cần giữ:** “missing region giống nền” là quan sát ảnh; không nói đã chứng minh backbone bỏ sót vì nguyên nhân đó. Với Fryum, không dùng lại mô tả cũ rằng response mạnh nhất chỉ lan trên toàn vật thể: bản đính chính xác nhận hotspot mạnh nhất ở **nền bên trái**.

**Tự kiểm tra:** mô tả từng failure bằng vị trí thật trên hình, metric đúng cấp và một giới hạn suy luận.

### Slide 14 - Ảnh hưởng RsCIN trên bốn category VisA

Đây là phép so sánh **cùng run, cùng ảnh, cùng anomaly map, trước và sau RsCIN**. Delta dương là metric tốt hơn sau RsCIN; delta âm là kém hơn. [E02Q]

| Category | Δ AUROC | Δ F1-max | Δ AP |
|---|---:|---:|---:|
| `pcb2` | +4,9500 | +1,3371 | +4,9193 |
| `pcb3` | -1,9406 | -1,3426 | -1,9673 |
| `pcb4` | +6,1980 | +10,1337 | +5,8446 |
| `fryum` | +1,1200 | +3,3765 | +0,5754 |

Đơn vị toàn bảng là điểm phần trăm. **Ba trong bốn category tốt hơn, một category xấu hơn**; không biến thành phát biểu 75% toàn VisA được cải thiện.

Trên `pcb3`, 85/100 ảnh anomaly giảm score và 72/101 ảnh normal tăng score. Số đếm cho thấy dịch chuyển đáng lo; bằng chứng phân tách kém hơn là **cả ba metric giảm**, vì chỉ giảm/tăng score tuyệt đối chưa tự chứng minh thứ hạng bị xấu.

**Giải thích hợp lý:** global feature có thể bị chi phối bởi cấu trúc board normal, khiến ảnh có lỗi nhỏ và ảnh normal vẫn là neighbor gần. Trộn score giữa chúng có thể làm yếu phân tách. Đồ án chưa kiểm định riêng từng cạnh graph để chứng minh mọi trường hợp đều do cơ chế này.

**Ablation và phân tích quan sát:** paper có ablation w/wo RsCIN. Đồ án có đối chiếu score trước/sau trên bốn category qua instrumentation; không phải một mô hình RsCIN mới được train. Hai việc đều kiểm tra tác động module nhưng phạm vi dữ liệu và kết luận khác nhau.

**Tự kiểm tra:** giải thích được vì sao RsCIN có ích trung bình trong paper vẫn có thể làm giảm metric ở một category cụ thể.

### Slide 15 - Top 1% so với maximum pooling

#### Giả thuyết và cách thử

Maximum chỉ dùng một cực đại, nên có thể nhạy với đỉnh nhiễu. Giả thuyết E04: mean các pixel score cao nhất giảm ảnh hưởng của cực đại và cải thiện image scoring.

Với map đã nội suy có $H\times W$ pixel, xếp giảm dần $A_{i,(1)}\ge\cdots$:

$$
c_i^{\max}=\max_{u,v}A_i(u,v),\qquad
c_i^{\mathrm{top1}}=\frac1T\sum_{t=1}^T A_{i,(t)},\qquad
T=\lceil0.01\cdot518\cdot518\rceil=2684.
$$

Top 1% là **2.684 pixel**, không phải 1% ảnh, không phải 1% patch token, không phải một vùng liên thông bắt buộc. Mean top pixel có thể lấy pixel từ nhiều hotspot rời nhau. Do map được nội suy từ patch grid, các pixel không phải quan sát độc lập mới.

Giữ map/backbone/LNAMD/MSM/split cố định, tạo bốn biến thể để phân biệt pooling và RsCIN:

| Biến thể | AUROC (%) | F1-max (%) | AP (%) |
|---|---:|---:|---:|
| Maximum, chưa RsCIN | 97,3864 | 96,7355 | 98,7258 |
| Maximum, sau RsCIN | 97,7340 | 97,4057 | 99,0528 |
| Top 1%, chưa RsCIN | 97,0227 | 96,8796 | 98,8246 |
| Top 1%, sau RsCIN | 96,7811 | 96,9747 | 98,7766 |

Primary comparison là top 1% so maximum **sau RsCIN**, macro qua 15 category. Delta lần lượt -0,9529 / -0,4310 / -0,2762 điểm. Trước RsCIN: AUROC -0,3638, F1-max +0,1441 và AP +0,0988. [E04]

#### Vì sao có thể tệ hơn?

Lỗi nhỏ thật có thể tạo đỉnh cao nhưng ít pixel; lấy mean top 1% làm nó giảm mạnh. Ngược lại, normal image có đáp ứng rộng theo vật thể có thể giữ mean top pixel tương đối cao. Xếp hạng giữa normal và anomaly vì vậy có thể xấu đi.

**Ví dụ số minh họa:** một ảnh anomaly có max 1,0 nhưng mean top vùng là 0,2; một ảnh normal có max 0,8 và mean top vùng là 0,6. Maximum xếp anomaly cao hơn normal; mean top vùng đảo thứ tự, dù score của từng ảnh đều không tăng.

Hai ca trên slide:

- `carpet/good/019`: FP → TN trong cả nhánh raw và RsCIN; giảm ảnh hưởng vùng biên mạnh hơn có lợi. Map còn activation khuếch tán, nên không nói chỉ có đúng một noisy pixel.
- `screw/good/003`: TN → FP trong cả hai nhánh; map đáp ứng gần toàn vật thể normal. Pooling không loại được broad structured activation.

**Vì sao score giảm vẫn thành FP?** Mỗi biến thể dùng threshold F1-max riêng. Ví dụ minh họa: score maximum 0,8 dưới threshold 0,9 là TN; score top vùng giảm còn 0,6 nhưng threshold của biến thể đó là 0,5 nên thành FP. Điều này không mâu thuẫn với $c_i^{\mathrm{top1}}\le c_i^{\max}$ trên cùng raw map. Không đem threshold của maximum áp sang top 1% rồi coi là so sánh operating point tương đương. [E04Q]

#### Phạm vi kết luận

Sau RsCIN, Image AUROC có 5 category tăng, 4 hòa và 6 giảm; median delta 0. `screw` giảm 9,7151 điểm, ảnh hưởng mạnh đến mean. Vì vậy kết quả âm tổng hợp không nghĩa mọi category giảm hoặc đã có bằng chứng thống kê về mọi quần thể công nghiệp. Các sign test thăm dò trong E04 không cho bằng chứng khác biệt có ý nghĩa theo ngưỡng thông thường.

Kết luận được phép: **top 1% không cải thiện baseline trên MVTec AD với cấu hình đã khóa**. Không được suy rộng thành mọi top-k pooling đều vô ích, hay đã cải thiện segmentation. Không đổi tỷ lệ sau khi nhìn kết quả chỉ để tìm một con số thuận lợi.

**Tự kiểm tra:** giải thích đủ ba ý: cùng map, ranking có thể đổi, threshold riêng có thể đổi.

### Slide 16 - Kết luận và giới hạn

Ba kết luận tương ứng ba loại bằng chứng:

1. **Tái lập:** đủ 27 category; pixel mean gần repo, lệch lớn nhất 0,0044 điểm sau làm tròn. Image mean còn sai khác lớn hơn.
2. **Chất lượng map:** có ca đúng vị trí và ca đáp ứng sai/mờ rộng; xếp hạng pixel tốt không tự động là segmentation gọn.
3. **Kiểm tra module/extension:** RsCIN khác hướng tác động theo category; top 1% có lợi cục bộ nhưng không cải thiện macro sau RsCIN.

**Giới hạn bắt buộc hiểu:** cùng lỗi lặp nhiều, normal appearance hiếm, khác orientation/scale, nền gây nhiễu, số ảnh lớn làm tăng chi phí, và không có kiểm định nguyên nhân môi trường bằng rerun tương ứng.

**Ranh giới đóng góp:** kết quả reproduction và negative result có giá trị kiểm tra tính tin cậy và hiểu phương pháp; không chuyển thành tuyên bố thuật toán mới hoặc kết quả state of the art của đồ án.

**Tự kiểm tra:** kết luận trong 30 giây mà vẫn nhắc được một kết quả đạt được và một giới hạn.

### Slide 17 - Tài liệu tham khảo

Mỗi tài liệu có một vai trò cụ thể:

| Tài liệu | Vai trò trong bài |
|---|---|
| Li et al., MuSc, ICLR 2024 | Phương pháp, công thức, main result và ablation gốc |
| Roth et al., PatchCore, CVPR 2022 | Đại diện normal-reference memory bank |
| Jeong et al., WinCLIP, CVPR 2023 | Đại diện zero-shot dựa trên text prompt |
| Zhou et al., AnomalyCLIP, ICLR 2024 | Đại diện học prompt trên dữ liệu phụ trợ |
| Radford et al., CLIP, ICML 2021 | Nguồn backbone và kiến thức image-text pretraining |
| Bergmann et al., MVTec AD, CVPR 2019 | Benchmark MVTec AD |
| Zou et al., SPot-the-Difference, ECCV 2022 | Công trình giới thiệu VisA và nghiên cứu liên quan |

Full citation và DOI được giữ trong [báo cáo, mục Tài liệu tham khảo](../2026-10-15-musc-final-report-draft-vi.md). Link repository chính thức và repository đồ án là hai nguồn khác nhau. Quyền truy cập repository đồ án cần kiểm tra trên tài khoản người nhận trước buổi trình bày; ghi chú này không thay đổi quyền truy cập.

**Tự kiểm tra:** gọi đúng CLIP là công trình nền, không phải baseline inference mà đồ án tự thiết kế.

## 4. Những phần trong paper cần biết khi bị hỏi sâu

### 4.1. Bằng chứng cho ba lựa chọn chính

| Câu hỏi | Nguồn paper | Kết luận đúng mức |
|---|---|---|
| Vì sao nhiều degree? | §3.1, Table 3, Appendix A.2.2 | Cân bằng kích thước lỗi và AC/AS; không phải tổ hợp thắng tuyệt đối mọi metric |
| Vì sao minimum 30% rồi mean? | §3.2, Table 4, Figure 7 | Giảm tác động normal appearance variation, tránh phụ thuộc một đối sánh; tỷ lệ 30% là lựa chọn thực nghiệm |
| Vì sao không loại luôn vài distance nhỏ nhất để tránh anomaly matching nhau? | Appendix A.2.3, Table 10 | Paper thử bỏ phần interval thấp nhất; overall không tốt hơn vì cũng đẩy score normal hiếm lên |
| Vì sao RsCIN cần neighborhood nhỏ? | §3.3, Table 5, Figure 8 | Giảm trộn ảnh ít liên quan; full graph/single window không luôn tốt bằng multi-window |
| Vì sao CLIP ViT-L/14@336? | Appendix A.2.1, Table 8 | Paper thử CLIP/DINO/DINOv2/Swin, chọn cân bằng kết quả; không chứng minh CLIP tốt nhất mọi nhiệm vụ |
| Dataset lớn xử lý ra sao? | Table 6-7, Appendix A.4 | Chia test pool giảm tài nguyên nhưng thay evidence và có thể giảm metric |
| Có kiểm tra ngoài hai benchmark? | Appendix A.5, Table 14 | Paper có BTAD; đồ án hiện tại không tái lập BTAD |
| Có few-shot extension không? | Appendix A.6 | Có thêm normal reference và MuSc+; không phải baseline zero-shot của đồ án |

Một số bảng paper có số khác nhẹ giữa main text và appendix, ví dụ VisA Pixel AUROC ở bảng ablation có 98,7 thay vì 98,8 ở Table 1. Khi trích số phải ghi đúng bảng, không trộn số của nhiều bảng thành một kết quả mới.

### 4.2. Hiệu năng paper và hiệu năng đồ án là hai nguồn

| Mốc | MVTec Image AUROC | MVTec Pixel AUROC | VisA Image AUROC | VisA Pixel AUROC |
|---|---:|---:|---:|---:|
| Paper Table 1 (%) | 97,8 | 97,3 | 92,8 | 98,8 |
| Repo ghim (%) | 97,77 | 97,11 | 92,57 | 98,71 |
| Đồ án E01/E02 (%) | 97,7340 | 97,1144 | 92,4001 | 98,7136 |

Paper có thể báo kết quả tốt hơn một baseline khác; đó là kết quả tác giả, không phải so sánh mới được đồ án tự chạy. So sánh khác protocol cần nêu nguồn dữ liệu phụ trợ, text prompt, normal reference, backbone và cách khai thác test set.

### 4.3. Compute, batch size và chia pool

Paper Table 6 báo 998,8 ms/ảnh và GPU peak 7.168 MB trên RTX 3090 ở setting được mô tả; đó không phải cam kết độ trễ nhận từng ảnh online. MuSc cần pool và mutual comparisons. Colab T4 của đồ án có hardware/software khác nên không so runtime trực tiếp như cùng phép đo.

E01 ghi 119,49 phút và E02 194,94 phút là **tổng thời gian subprocess thành công**, không bao gồm tải dữ liệu, setup, attempt bị ngắt và tổng hợp. Không gọi đây là toàn thời gian thực hiện đồ án.

`divide_num>1` chia pool thành các subset: giảm $N$ trong từng lần MSM nhưng thay tập tham chiếu. Giảm batch size giữ cùng pool có thể giúp phần trích feature; chia pool thay đổi cả bài toán inference. Cần phân biệt hai cách giảm tài nguyên.

### 4.4. Hai thay đổi kỹ thuật của reproduction

1. **Import compatibility:** ưu tiên local package `datasets` của MuSc trước Hugging Face `datasets`, thêm `datasets/__init__.py`. Mục đích là import đúng code dataset.
2. **Memory cleanup:** giải phóng tensor dư và tránh bản sao NumPy không cần thiết sau tạo map để giảm CPU RAM.

Semantic inspection không thấy thay công thức chấm điểm. Tuy nhiên không có same-category patched/unpatched A/B trong cùng môi trường cho memory patch. “Bottle clean-session check” từng tái sử dụng kết quả đã thành công nên không phải một lần rerun mới chứng minh equivalence. [E01]

### 4.5. Hạn chế của các hình được chọn

Các hình là ca chẩn đoán được chọn theo quy tắc, gồm trường hợp tốt, xấu và score cực trị. Không phải mẫu ngẫu nhiên đại diện để nói “đa số trường hợp ngoài thực tế đều như hình”.

`whole_norm` tạo một thang hiển thị chung trong phạm vi run/category tương ứng; không phải calibrated scale giữa mọi category, mọi run hoặc mọi score variant. Overlay là minh họa vị trí. Raw map mới là đầu vào chính để tính benchmark metrics; số từ heatmap đã lưu cần ghi rõ nguồn.

## 5. Câu hỏi phản biện và ý trả lời

### 5.1. Không nhãn thì làm sao biết normal?

MuSc không xác nhận normal bằng nhãn. Nó dựa vào feature pretrained và mức lặp lại của pattern giữa nhiều ảnh. Cấu trúc tìm được nhiều đối sánh gần nhận score thấp. Đây là giả định thống kê có thể sai, không phải cách thay nhãn bằng một bộ luật chắc chắn.

### 5.2. Nếu phần lớn ảnh đều lỗi thì có chạy được không?

Có thể, nếu các ảnh vẫn chứa nhiều vùng normal và lỗi không lặp lại quá giống nhau. Tỷ lệ ảnh lỗi cao không tự động phá phương pháp. Nguy hiểm trực tiếp hơn là **cùng pattern lỗi lặp đủ nhiều** để trở thành đối sánh gần cho nhau. Không có tỷ lệ an toàn phổ quát được đồ án xác định.

### 5.3. Vì sao không nearest neighbor toàn dataset cho nhanh?

Một đối sánh tình cờ với patch lỗi khác có thể làm anomaly score rất nhỏ. MSM giữ một khoảng cách trên mỗi ảnh rồi cần hỗ trợ từ nhiều ảnh trong interval, giảm phụ thuộc vào một patch duy nhất. Table 4 của paper kiểm tra các phép tổng hợp thay thế.

### 5.4. Nếu normal có nhiều pose thì 30% giải quyết hoàn toàn chưa?

Chưa. Nó cho phép query chỉ cần một nhóm ảnh tương thích, nhưng nếu pose normal hiếm đến mức thiếu đối sánh, score vẫn có thể cao. Feature của backbone cũng không hoàn toàn bất biến với orientation và scale. Paper Appendix A.3 thảo luận giới hạn này.

### 5.5. Đổi 30% hoặc RsCIN k có thể tốt hơn không?

Có thể trên một tập cụ thể, nhưng đồ án giữ cấu hình official để đánh giá tái lập. Chọn hyperparameter sau khi xem test labels sẽ đổi câu hỏi thực nghiệm và gây nguy cơ overfitting benchmark. Muốn nghiên cứu tuning phải có thiết kế validation và báo cáo riêng.

### 5.6. Có phải MuSc dựa trên attention map của CLIP?

Không. Map được tính từ khoảng cách aggregated patch feature giữa các ảnh, rồi kết hợp và nội suy. Self-attention giúp tạo feature trong backbone; attention weights không phải anomaly map được hiển thị.

### 5.7. Vì sao dùng class token mà không dùng text để sửa score?

Class token cung cấp image similarity trong chính pool; RsCIN dùng thông tin đó để kết hợp anomaly score. Không cần mô tả lỗi bằng ngôn ngữ, nhưng similarity của global appearance có thể không khớp trạng thái lỗi nhỏ.

### 5.8. Có thể dùng trực tiếp cho một ảnh upload mới không?

Theo protocol gốc cần pool nhiều ảnh cùng category. Dùng reference pool cố định để nhận ảnh mới là adaptation có thể thiết kế, nhưng cần đánh giá lại; không tự xem kết quả benchmark transductive là bằng chứng cho vận hành từng ảnh online.

### 5.9. Tối thiểu cần bao nhiêu ảnh?

Không có con số bảo đảm chất lượng chung. Về code ghim, MSM cần $\lfloor0,3(N-1)\rfloor>0$, tương ứng ít nhất 5 ảnh; RsCIN VisA dùng window đến 9 nên cần ít nhất 9 để thao tác neighbor hợp lệ. Đây chỉ là điều kiện kỹ thuật cần, **không phải khuyến nghị số mẫu đủ tin cậy**. Pool cần đủ normal diversity và không bị lặp lỗi chi phối.

### 5.10. Tại sao Image AUROC thấp trong khi Pixel AUROC cao?

Hai cấp đo khác nhau. Nhiều pixel normal nền được xếp thấp giúp pixel ranking tốt; nhưng một hotspot mạnh trên ảnh normal có thể đẩy max image score cao. Ngược lại, lỗi nhỏ có ranking pixel khá tốt nhưng max không đủ tách ảnh. RsCIN còn có thể thay thứ hạng cấp ảnh.

### 5.11. Có cần threshold khi tính AUROC/AP không?

Không chọn một threshold vận hành duy nhất trước khi tính. Chúng đánh giá ranking qua nhiều ngưỡng. F1-max cũng quét ngưỡng nhưng chọn giá trị F1 tốt nhất; FP/TN từng ảnh chỉ có nghĩa khi nêu operating threshold.

### 5.12. Có nên báo accuracy không?

Accuracy tại một threshold có thể bị tỷ lệ normal/anomaly chi phối và không thể hiện localization. Đồ án dùng metric theo paper để so sánh. Khi triển khai cần thêm yêu cầu cụ thể như recall tại FPR mục tiêu, nhưng chưa có thực nghiệm triển khai để báo số đó.

### 5.13. Top 1% có làm giảm recall vì lỗi nhỏ không?

Đó là cơ chế khả dĩ: lỗi rất nhỏ có peak cao nhưng mean top vùng bị pha với vùng ít bất thường. E04/E04Q có ca rank giảm hoặc quyết định xấu đi, nhưng không chứng minh đó là nguyên nhân duy nhất cho toàn bộ macro delta. Broad normal activation và RsCIN cũng tương tác với score mới.

### 5.14. Top 1% có ý nghĩa gì nếu kết quả âm?

Nó kiểm tra một giả thuyết rõ với đúng yếu tố thay đổi, giữ map cố định và báo đủ category. Kết quả xác định giới hạn của trực giác chống peak nhiễu trong cấu hình này; không cần tô thành cải tiến để có giá trị phân tích.

### 5.15. Chỉ bốn category RsCIN có đủ kết luận không?

Đủ để nói tác động không đồng nhất và đưa ra phản ví dụ `pcb3` trong run này. Không đủ để ước lượng tỷ lệ category được lợi ngoài mẫu hay kết luận RsCIN kém trên toàn VisA. Cũng không đủ để giải thích nguyên nhân chênh với repo.

### 5.16. MuSc hơn PatchCore không?

Không có kết luận tuyệt đối. Paper cho thấy MuSc mạnh trong zero-shot và cạnh tranh với một số few/full-shot methods; PatchCore full-shot có normal reference riêng và kết quả rất mạnh. Cần so theo điều kiện sử dụng dữ liệu và metric, không chỉ theo tên phương pháp.

### 5.17. Có thể áp dụng y khoa hoặc sản phẩm mới không?

Category sản phẩm mới có thể thử theo protocol cùng category, nhiều ảnh chưa gán nhãn. Chuyển sang y khoa là thay domain; đồ án không cung cấp bằng chứng cho hiệu quả hoặc chẩn đoán ở domain đó. Cần dataset, ground truth và đánh giá phù hợp trước khi nhận định.

### 5.18. Đồ án đã kiểm chứng những gì, chưa kiểm chứng những gì?

Đã có run đủ hai benchmark, log/config/metric lưu lại, đối chiếu bảng tham chiếu, định tính và controlled pooling comparison. Lần chuẩn bị ghi chú này đối chiếu tài liệu/code, không rerun mô hình. Chưa có matched official-environment rerun, A/B memory patch đầy đủ, kiểm định triển khai online hay generalization sang domain mới.

## 6. Tờ ôn nhanh trước buổi trình bày

### 6.1. Mười ý cần nhớ

1. Input: **một pool nhiều ảnh chưa gán nhãn cùng category**.
2. Không target training, không text prompt; backbone CLIP đã pretrain.
3. Cốt lõi: normal pattern có nhiều đối sánh; anomaly pattern thường ít đối sánh.
4. Backbone: input 518, patch 14, lưới 37 × 37, bốn stage.
5. LNAMD: ba mức neighborhood 1/3/5 trên token.
6. MSM: min trong từng ảnh khác → mean 30% nhỏ nhất → mean 12 nhánh.
7. Map: reshape/upsample; image score: maximum rồi RsCIN.
8. RsCIN chỉ đổi image score, không đổi map.
9. Tái lập: 15 + 12 = 27 category; pixel mean lệch lớn nhất 0,0044 điểm.
10. Giới hạn: RsCIN có phản ví dụ; top 1% sau RsCIN giảm Image AUROC macro 0,9529 điểm.

### 6.2. Những con số đáng nhớ

| Con số | Ý nghĩa chính xác |
|---|---|
| 518 × 518 | Input inference của cấu hình tái lập |
| 37 × 37 = 1.369 | Lưới patch và số patch trên mỗi ảnh/stage |
| 6, 12, 18, 24 | Block trích feature; config lưu index 5, 11, 17, 23 |
| 30% | Phần khoảng cách nhỏ nhất trong vector liên ảnh của mỗi query patch |
| 15 / 12 / 27 | Số category MVTec / VisA / tổng |
| 1.725 / 2.162 | Số ảnh test quantitative MVTec / VisA |
| 0,0044 điểm | Sai lệch tuyệt đối pixel mean lớn nhất so với repo, làm tròn |
| 0,0360 điểm | Sai lệch tuyệt đối mean lớn nhất trong bảy metric MVTec |
| -0,2622 điểm | Delta Image F1-max mean VisA so repo |
| +10,1337 điểm | F1-max `pcb4` sau - trước RsCIN |
| 2.684 | Số pixel lấy mean trong top 1% của map 518 × 518 |
| -0,9529 điểm | Δ Image AUROC macro top 1% - max sau RsCIN |

### 6.3. Chuỗi giải thích một phút

MuSc phát hiện và định vị bất thường công nghiệp bằng một tập nhiều ảnh chưa gán nhãn cùng category. Phương pháp giữ nguyên backbone CLIP đã pretrain và không dùng text prompt. Patch token được tổng hợp ở ba mức neighborhood bằng LNAMD, sau đó MSM tìm patch gần nhất trong từng ảnh khác và lấy mean 30% distance nhỏ nhất. Pattern normal thường có đủ đối sánh để nhận score thấp; pattern lỗi hiếm thường nhận score cao. Kết hợp các stage/degree tạo anomaly map; maximum tạo image score rồi RsCIN kết hợp score theo global image similarity. Đồ án đã chạy đủ 27 category của hai benchmark; pixel mean gần bảng repo nhưng image score còn sai khác. Các hình cho thấy AUROC cao vẫn có thể đi cùng map chưa gọn. Top 1% sửa một số false positive nhưng giảm kết quả macro sau RsCIN, nên chưa phải cải tiến cho cấu hình này.

### 6.4. Tự kiểm tra không nhìn tài liệu

- Vẽ lại bằng lời hai nhánh patch token và class token của slide 5.
- Tính một ví dụ 10 khoảng cách và chọn đúng ba distance nhỏ nhất.
- Giải thích ba phép tổng hợp khác nhau của MSM.
- Phân biệt input 518 với tên pretrained 336.
- Giải thích score, xác suất, threshold và F1-max là bốn khái niệm khác nhau.
- Đọc đúng đơn vị và mẫu số của mọi con số trên slide 10-15.
- Chỉ đúng lỗi, GT và hotspot trên bốn hình slide 12-13.
- Giải thích TN → FP của top 1% mà vẫn giữ bất đẳng thức raw mean top ≤ raw max.
- Nêu được một điều bằng chứng xác nhận và một điều mới chỉ là giả thuyết.

Mức nắm bài đủ để bảo vệ không phải nhớ toàn bộ bảng, mà là giải thích được mỗi phép tính dùng nguồn thông tin nào, giải quyết vấn đề gì, thất bại khi nào và bằng chứng nào hỗ trợ kết luận đang nói.
