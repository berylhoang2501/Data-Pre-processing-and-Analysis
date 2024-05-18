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

**Biểu đồ**

<img width="753" alt="Ảnh màn hình 2024-05-18 lúc 10 36 09" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/c79a6c5a-5b6f-403f-b9ee-dcc1f132d663">

**Kiểm định giả thuyết**

Ví dụ: Dùng kiểm định ANOVA để xem có mối tương quan giữa các biến Pclass, biển Sex và biến Survived?

- Các biến độc lập (input): Pclass và Sex
  
- Biến phụ thuộc (output): Survived

Nếu kết quả kiểm định xét có mối tương quan thì là do ngẫu nhiên hay có ýnghĩa thống kê?

**Phân tích dữ liệu cần lưu ý**

- Không có kỹ năng phân tích đúng.
  
- Sử dụng các công cụ sai để phân tích dữ liệu. Ví dụ: sử dụng z-score khi dữ liệu không có phân phối chuẩn. (lúc nào cũng phải xem xét lý do vì sao lại lựa chọn công cụ đó để phân tích)
  
 - Để bias ảnh hưởng đến kết quả. (sự thiên vị, cái lớn áp đảo cái nhỏ)
 
-  Không tìm ra ý nghĩa thống kê. (ý nghĩa tìm ra có đúng trên tổng thể hay không)

- Phát biểu không chính xác null hypothesis và alternate hypothesis.

- Sử dụng graph và chart không chính xác, gây hiểu lầm. 

### 4. Quy trình phân tích dữ liệu

**Quy trình Data Analysis**

<img width="776" alt="Ảnh màn hình 2024-05-18 lúc 10 56 59" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/50e489e8-580b-4b94-9b9f-7121287aac28">

<img width="811" alt="Ảnh màn hình 2024-05-18 lúc 11 00 54" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/74e06432-9810-4446-aa5c-ad6e78a3ccc2">

 1. Business Understanding (Xác định vấn đề kinh doanh cần giải quyết) -> 2. Data Requirements -> 3. Data Collection -> *4. Data Pre-processing (chiếm 60 đến 80% quy trình) ->  *5. Exploratory Data Analysis (EDA) -> *6. Modeling & Algorithms -> 7. Data Product -> 8. Communication
