# Data-Pre-processing-and-Analysis
## Bài 1: Tổng quan Data Analysis
### 1. Giới thiệu

thu thập dữ liêụ -> tiền xử lý dữ liệu -> sác xuất thống kê -> machine learning 

clustering (phân cụm), classification (phân loại)

- ôn lại phần thống kê và vẽ biểu đồ 

### 2. Một số kỹ thuật phân tích dữ liệu

**Thống kê mô tả (descriptive analysis)**

- mô tả tóm tắt các đặc điểm cơ bản của dữ liệu

<img width="570" alt="Ảnh màn hình 2024-05-18 lúc 09 06 07" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/771580c3-185f-47e2-b16b-0c38ce99c086">

**Thống kê suy luận (Inferential Statistics)**

- đc học trg môn math

- Mục đích là suy luận (ước lượng) ra các tham số của quần thể (population) dựa trên các tham số của mẫu (sample).

- Thống kê suy luận sử dụng kiểm định giả thuyết để xem có sự khác biệt giữa tham số của mẫu và của quần thể. Sự khác biệt đó là có ý nghĩa thống kê hay chỉ do ngẫu nhiên.

- Một số kiểm định giả thuyết như: t-test, chi- square, ANOVA, ...

-  Sử dụng kiểm định giả thuyết, có thể suy luận rằng mức độ hài lòng của mẫu có thể đại diện cho mức độ hài lòng của toàn bộ khách hàng
(quần thể), với một hệ số tin cậy (thường chọn 0.95 ~ khoảng tin cậy 95%).

**Trực quan hóa dữ liệu (Data Visualization)**

- trục x,y sẽ đại diện cho những gì,..

**Phân tích tương quan (Correlation Analysis)**

- tìm mối quan hệ giữa 2 biến số

- Một trong những phương pháp phổ biến để đo lường mức độ tương quan là sử dụng hệ số tương quan Pearson.

- giá trị đo lường cho tương quan pearson trả kq về từ (-1,1). càng gần 1 và -1 thì mức độ tương quan càng mạnh (càng gần -1 là tương quan nghịch, càng gần 1 là tương quan thuận cái này tăng thì cái kia tăng)

ví dụ tương quan thuận: số năm kinh nghiệm tăng, lương  tăng 

ví dụ tương quan nghịch: giá xe càng cao thì càng tiết kiệm xăng

**Phân tích tương quan (Correlation Analysis)**

- là tiếp theo của phân tích tương quan 

- Phân tích hồi quy khám phá mối quan hệ giữa một biến phụ thuộc (dependent) với một (nhiều) biến độc lập (independent).
  
- Phân tích hồi quy đo lường sức mạnh mối quan hệ giữa các biến. Từ đó, sử dụng chúng vào mô hình để dự đoán.

**Phân tích tình huống/ kịch bản (Scenario Analysis)**

- Phân tích tình huống/ kịch bản là phân tích các sự kiện có thể xảy ra trong tương lai với các kết quả thay thế.

**Khai phá dữ liệu (Data Mining)**

Data mining sử dụng các kỹ thuật để: Xây dựng các mô hình, Tìm kiếm các quy luật, hay xu hướng từ dữ liệu.

***Một số kỹ thuật:***
- Clustering (phân cụm): Phân loại các đối tượng thành các nhóm có tính chất tương đồng. (mô hình tự đi tìm)

Ví dụ: Phân loại khách hàng dựa trên hành vi mua sắm để tạo các nhóm tiềm năng khách hàng ?

- Classification (phân loại): Xây dựng mô hình để phân loại dữ liệu vào các lớp đã biết trước. (mình tự xác định)

Ví dụ: Dự đoán xem một email là mail rác (spam) hay không dựa trên nội dung và thông tin liên quan.

- Association Rule Mining (Khám phá quy luật kết hợp): Tìm ra các mối quan hệ và quy luật giữa các biến. Ví dụ: Phát hiện rằng khách hàng mua sản phẩm Athường đi kèm với mua sản phẩm B.

- Outlier Detection (Phát hiện ngoại lệ): Xác định các dữ liệu ngoại lệ hoặc không tuân theo quy luật chung. Ví dụ: Phát hiện giao dịch tài chính bất
thường trong một khoảng thời gian. -> xác định nguyên nhân của outliner (gõ nhầm? thực tế có?..)

tứ phân vị iqr = q3 - q1

**A/B Testing**

- Phân tích A/B testing là một phương pháp thống kê được sử dụng trong nghiên cứu thị trường và quảng cáo để đánh giá hiệu suất của hai biến Avà B(các biến thử nghiệm) để xét xem biến thử nghiệm nào "tốt hơn".

**Phân tích thăm dò (EDA - Exploratory Data Analysis)** 

- EDA là quá trình khám phá thăm dò dữ liệu để hiểu rõ các đặc điểm chính, mối quan hệ và xu hướng trong dữ liệu.

- EDA sử dụng các kỹ thuật sau: Trực quan hóa dữ liệu, Thống kê mô tả, Thống kê suy luận

### 3. Một số công cụ phân tích dữ liệu

**Bảng**

- Ví dụ: Tạo bảng phân phối tần suất (frequency distribution table)

- sử dụng group by trong pandas

### 4. Quy trình phân tích dữ liệu
