# Data-Pre-processing-and-Analysis
## Bài 1: Tổng quan Data Analysis (18/5/2024)
### 1. Giới thiệu

thu thập dữ liêụ -> tiền xử lý dữ liệu -> sác xuất thống kê -> machine learning 

clustering (phân cụm), classification (phân loại)

- ôn lại phần thống kê và vẽ biểu đồ 

### 2. Một số kỹ thuật phân tích dữ liệu

**Thống kê mô tả (descriptive analysis)**

- mô tả tóm tắt các đặc điểm cơ bản của dữ liệu

<img width="570" alt="Ảnh màn hình 2024-05-18 lúc 09 06 07" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/771580c3-185f-47e2-b16b-0c38ce99c086">

trung vị (median)  là một thống kê chỉ trung tâm của một tập hợp dữ liệu, được xác định bằng cách sắp xếp các giá trị theo thứ tự tăng dần và chọn giá trị ở giữa. Nếu tập dữ liệu có số lượng phần tử là số lẻ, trung vị là giá trị ở vị trí giữa. Nếu tập dữ liệu có số lượng phần tử là số chẵn, trung vị là trung bình của hai giá trị ở giữa. Trung vị được sử dụng để xử lý dữ liệu bị thiếu (missing values). Khi dữ liệu có các giá trị bị thiếu, bạn có thể thay thế các giá trị này bằng trung vị của các giá trị còn lại trong tập dữ liệu để giảm thiểu ảnh hưởng của các giá trị ngoại lệ (outliers). (giả sử có 1 dữ liệu bị outlier quá lớn thì khi dùng thống kê mean thì dữ liệu sẽ bị ảnh hưởng theo, trong khi đó nếu dùng trung vị thì sẽ không bị ảnh hưởng)

Yếu vị (mode) là giá trị xuất hiện nhiều nhất trong một tập hợp dữ liệu. Một tập dữ liệu có thể có một yếu vị (unimodal), hai yếu vị (bimodal), hoặc nhiều yếu vị (multimodal). Nếu không có giá trị nào lặp lại, tập dữ liệu đó không có yếu vị. Yếu vị có thể được sử dụng để xử lý dữ liệu bị thiếu (missing values). Yếu vị là lựa chọn tốt khi bạn làm việc với dữ liệu phân loại (categorical data). (ví dụ điền giá trị phổ biến nhất trong tập dữ liệu)

Phương sai (variance) là một thước đo thống kê cho thấy mức độ phân tán của các giá trị trong một tập dữ liệu xung quanh giá trị trung bình của nó. 

Độ lệch chuẩn (standard deviation) là một thước đo thống kê biểu thị mức độ phân tán của các giá trị trong một tập dữ liệu so với giá trị trung bình của nó. Nó là căn bậc hai của phương sai và cung cấp một cách đo lường dễ hiểu hơn về sự phân tán vì nó có cùng đơn vị với dữ liệu ban đầu.

Định Nghĩa Khoảng Tứ Phân Vị  (Interquartile Range - IQR): Khoảng tứ phân vị (IQR) là một thước đo thống kê biểu thị mức độ phân tán của dữ liệu, được tính bằng cách lấy hiệu số giữa giá trị phân vị thứ ba (Q3) và giá trị phân vị thứ nhất (Q1). Nó đại diện cho khoảng mà 50% dữ liệu trung tâm nằm trong đó, loại bỏ ảnh hưởng của các giá trị ngoại lệ.

<img width="524" alt="Ảnh màn hình 2024-05-22 lúc 02 17 30" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/ee342c53-e3fa-4e37-9e2f-b25b4484868f">

**Thống kê suy luận (Inferential Statistics)**

- đc học trg môn math

- Mục đích là suy luận (ước lượng) ra các tham số của quần thể (population) dựa trên các tham số của mẫu (sample).

- Thống kê suy luận sử dụng kiểm định giả thuyết để xem có sự khác biệt giữa tham số của mẫu và của quần thể. Sự khác biệt đó là có ý nghĩa thống kê hay chỉ do ngẫu nhiên.

- Một số kiểm định giả thuyết như: t-test, chi-square, ANOVA, ...

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

**Phân tích hồi quy (Regression Analysis)**

- là tiếp theo của phân tích tương quan 

- Phân tích hồi quy khám phá mối quan hệ giữa một biến phụ thuộc (dependent) với một (nhiều) biến độc lập (independent).
  
- Phân tích hồi quy đo lường sức mạnh mối quan hệ giữa các biến. Từ đó, sử dụng chúng vào mô hình để dự đoán.

- Biến phụ thuộc (Dependent Variable) là biến đầu ra: Còn gọi là biến kết quả (outcome variable), là biến mà chúng ta muốn nó hiển thị kết quả đầu ra của mô hình (ví dụ biến Survived sẽ đại diện cho việc hành khách sống sót hay không sau thảm họa Titanic dựa trên hai giá trị là 0: Không sống sót và 1: Sống sót). 

- Biến độc lập (Independent Variable) là các biến đầu vào: Còn gọi là biến dự báo (predictor variable), là các biến được sử dụng để dự đoán hoặc giải thích sự thay đổi trong biến phụ thuộc. Chúng độc lập và không bị ảnh hưởng bởi các biến khác trong mô hình hồi quy. (ví dụ Biến độc lập là Pclass, Sex, Age, SibSp, Parch, Fare, và Embarked (chúng ta sử dụng các biến này để dự đoán giá trị của biến Survived ).

**Phân tích tình huống/ kịch bản (Scenario Analysis)**

- Phân tích tình huống/ kịch bản là phân tích các sự kiện có thể xảy ra trong tương lai với các kết quả thay thế.

**Khai phá dữ liệu (Data Mining)**

Data mining sử dụng các kỹ thuật để: Xây dựng các mô hình, Tìm kiếm các quy luật, hay xu hướng từ dữ liệu.

***Một số kỹ thuật:***
- Clustering (phân cụm):  các đối tượng thành các nhóm có tính chất tương đồng. (mô hình tự đi tìm)

Ví dụ:  khách hàng dựa trên hành vi mua sắm để tạo các nhóm tiềm năng khách hàng ?

- Classification (): Xây dựng mô hình để  dữ liệu vào các lớp đã biết trước. (mình tự xác định)

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

data intergration là tập hợp dữ liệu từ nhiều nguồn về thành 1 

noise identification (outliner, sai dữ liệu, dữ liêụ không nhất quán ví dụ như khác đơn vị và không có cái nhìn chung)

 - 1. Business Understanding (Xác định vấn đề kinh doanh cần giải quyết) -> 2. Data Requirements -> 3. Data Collection -> *4. Data Pre-processing (chiếm 60 đến 80% quy trình) ->  *5. Exploratory Data Analysis (EDA) -> *6. Modeling & Algorithms -> 7. Data Product -> 8. Communication

dữ liệu để train -> dữ liệu để test -> score (nếu điểm score thấp có nghĩa là mô hình học chưa tốt, điều này có thể phụ thuộc vào nhiều lý do và đây là lúc cần rà soát lại các bước trong quy trình để tìm ra ở bước nào đã gây ảnh hưởng đến việc học mô hình)

https://www.kaggle.com

https://www.w3schools.com
cần xem lại , matplotlib

## Bài 2: Tổng quan Data Pre-processing (19/5/2024)
### 1. Giới thiệu

Dữ liệu trong thể giới thực thường là: 

- Dữ liệu sai. 

- Dữ liệu thiếu.

- Dữ liệu trùng lặp.

- Dữ liệu không nhất quán. (ví dụ nhập nơi ở có người nhập tphcm, tp hồ chí minh,..)

- Dữ liệu có các giá trị ngoại lai (outlier).

- Dữ liệu chưa được chuẩn hóa. (đo = skew, > 0 là lệch phải, <0 là lệch trái) . biểu đồ histogram gần xét phân phối chuẩn, lệch trái, lệch phải

cần phải có trung vị trong trường hợp giá trị trung bình bị outlier

lý tưởng nhất là đưa dữ liệu về dạng không bị lệch thì mô hình sẽ học tốt nhất, không bị nhiễu

**Tiền xử lý dữ liệu - Làm sạch dữ liệu**

### 2. Quy trình Data Pre-processing

**Import thư viện**
- NumPy là package hỗ trợ cho việc tính toán với Python.

- Pandas là package hỗ trợ việc thao tác và phân tích dữ liệu

- Matplotlib là package vẽ biểu đồ 2D của Python

- Seaborn là package trực quan hóa dữ liệu Python dựa trên matplotlib, cung cấp giao diện cấp cao để vẽ biểu đồ thống kê hấp dẫn hơn.

**Đọc/tích hợp dữ liệu, lựa chọn thuộc tính**

**Kiểm tra dữ liệu thiếu (missing value), nhiễu (noise), ngoại lệ (outlier), trùng (duplicate), không nhất quán (inconsistencies)**

**Kiểm tra dữ liệu  (categorical data)**

**Chuẩn hóa dữ liệu (Data standardizing)**

**Kỹ thuật tính năng (feature engineering)**

**Chuyển đổi dữ liệu (transformation)**

**Chia dữ liệu (Data splitting)**

## Bài 3: Data Pre-processing (19/5/2024)

### 1. Giới thiệu

**Một số bước trong data pre-processing**

- Làm sạch dữ liệu: Xử ýl dữ liệu bị thiếu, dữ liệu có các giá trị ngoại lệ.

- Chuẩn hóa dữ liệu: Đảm bảo rằng dữ liệu ở định dạng chung và có cùng đơn vị đo lường.

- Tạo mới đặc trưng (feature engineering):

- Tạo ra các đặc trưng mới, rút trích đặc trưng đểcải thiện khả năng dự đoán của mô hình.

**Kiểu dữ liệu cơ bản**
- Kiểu chuỗi (string): Ví dụ: họ tên nhân viên; tên sản phẩm, địa ch khách hàng...

- Kiểu số (numeric): số nguyên và số thực. Ví dụ: Tuổi kiểu số nguyên; Lương kiểu số thực.

- Kiểu ngày (datetime): Ví dụ: ngày sinh; ngày mua hàng...

- Kiểu Boolean: true/false; 1/0. Ví dụ: true:sống; false: chết...

### 2. Các bước thực hiện

![Ảnh màn hình 2024-05-19 lúc 10 48 40](https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/e9331626-5049-44fb-93d5-080dfbaa81b4)

#### ***A. Đọc dữ liệu và xem thông tin cơ bản***
**Đọc dữ liệu**

- Tập tin (.CsV, .txt, json, XIsx, ...): sử dụng pandas

**Tích hợp dữ liệu**

- Nối các dataframe lại: sử dụng pandas.concat()

- Trộn các dataframe lại: sử dụng pandas.merge)

thư viện glob: cho phép tự động đọc nhiều tập tin

<img width="511" alt="Ảnh màn hình 2024-05-24 lúc 20 37 37" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/a457f6bf-de6a-4d2c-85df-846d410167be">

ví dụ: đọc tất cả các tệp CSV trong thư mục data có tên bắt đầu bằng iris..

**Xem thông tin cơ bản**

- shape, info, dtypes, head, tail, columns

**Xác định các thuộc tính**

- hiểu rõ về tập dữ liệu (Cần biết rõ về kiểu dữ liệu và ý nghĩa của các cột, Cột nào là cột hữu ích, Cần biết các mối quan hệ giữa các cột (nếu có)

- Xác định biến đầu vào (Input/ Independent) và biến đầu ra (Output/ Dependent).

- Xácđịnh kiểu dữ liệu của các biến: (xác định biến đó là biến  hay biến số, biến chuỗi, biến ngày, biến boolean..)

biến số (biến liên tục thường là số nguyên và biến rời rạc thường là số thực)

- biến liên tục: đo chiều cao của học sinh và nhận được các giá trị như 150.2 cm, 160.5 cm, 172.8 cm, thì chiều cao có thể là bất kỳ số nào trong khoảng từ, ví dụ, 150 cm đến 180 cm, bao gồm cả các số thập phân như 155.55 cm hay 167.8 cm.
- biến rời rạc: Nếu chúng ta đếm số học sinh trong một lớp, chúng ta có thể có các giá trị như 25 học sinh, 30 học sinh, 35 học sinh. Không thể có 25.5 hay 30.7 học sinh vì số học sinh phải là một số nguyên.

biến phân loại (dùng nghiệp vụ, chọn ra các thuộc tính có ý nghĩa để phân tích), biến phân loại quan trọng nhất là ý nghĩa thống kê của nó, kqt nó là biến số hay chuỗi

<img width="293" alt="Ảnh màn hình 2024-05-24 lúc 20 51 15" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/92f2e809-9b55-42ad-88f9-227951a2a681">

<img width="770" alt="Ảnh màn hình 2024-05-19 lúc 11 06 09" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/69ecaf15-6555-4c79-be19-f6fab5e180cb">

- select_dtypes: xác định các biến thuộc cùng kiểu dữ liệu (ví dụ: xác định các biến thuộc kiểu dữ liệu object)

- trong trg hợp có nhiều biến thì mới dùng cách này

  <img width="943" alt="Ảnh màn hình 2024-05-24 lúc 21 03 11" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/7c736d60-b3ad-4961-ad75-d705fa106b97">

hình ảnh: chọn lần lượt các cột, nếu giá trị đặc biệt của cột đó dưới 5 (thường là biến phân loại) thì in ra còn trên 5 thì đếm số lượng thôi

lưu ý: không phải biến kiểu object nào cũng là biến phân loại

<img width="923" alt="Ảnh màn hình 2024-05-24 lúc 21 10 22" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/a2a5de1e-fe96-4e49-8edd-964f94784ec9">

hình ảnh: nếu unique value nhỏ hơn = 10 (bién phân loại) thì in ra, > thì in số lượng thôi

**Thống kê mô tả**

- Thống kê các biến số: count, mean, std, min, 25%, 50% (median), 75%, min, max.

-  Thống kê các biến phân loại: count, unique, top, freq.

**Chuyển đổi kiểu dữ liệu**

<img width="507" alt="Ảnh màn hình 2024-05-25 lúc 05 54 28" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/193faece-51f9-4958-a629-15e6530bc8be">

**Xử lý dữ liệu trùng lặp**

- Phát hiện dữ liệu trùng lặp bằng hàm .duplicated().sum() và .duplicated().any()


#### ***B. Làm sạch dữ liệu***

- Loại bỏ dữ liệu trùng lặp bằng hàm drop_duplicate()

**Xử lý dữ liệu thiếu**

<img width="976" alt="Ảnh màn hình 2024-05-22 lúc 18 22 03" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/a3f877db-e345-452e-b26a-1a48b6797bf2">

Xoá dòng

Xoá cột

- Mean/ Mode/ Median Imputation

<img width="938" alt="Ảnh màn hình 2024-05-25 lúc 06 19 47" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/a66bcdb5-b352-46ca-8141-0dc8999981f1">

ví dụ 2: điền bằng tuổi trung bình theo pclass (điền tuổi khác nhau đối với các hành khách ở các hạng ghế khác nhau), dùng câu lệnh transform

**Dữ liệu không nhất quán**

- ví dụ: Dữ liệu không nhất quán về đơn vị

**Phát hiện ngoại lệ (Outlier)**

- Ngoại lệ có hai loại: đơn biến (Univariate) và đa biến (Multivariate)

<img width="944" alt="Ảnh màn hình 2024-05-25 lúc 06 23 23" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/4f550207-7371-4bf8-868e-87a340442e8a">

các biến xét riêng lẽ không có ngoại lệ nhưng khi kết hợp với nhau thì có khả năng có ngoại lệ 

- xét ngoại lệ trên 1 biến -> vẽ boxplot

- xét ngoại lệ trên 2 biến -> vẽ scatter plot

- Cách phát hiện ngoại lệ: Thường được sử dụng phổ biến là trực quan hóa (boxplot, histogram, scatterplot), Áp dụng một số quy tắc như z-score ngoài 3 độ lệch chuẩn; tứ phân vị.

- Xử lý ngoại lệ: Hầu hết các cách xử ýl ngoại ệl tương tự như các cách xử ýl với dữ liệu bị thiếu Như xóa mẫu, biến đổi chúng, binning, tạo các riêng biệt, thay thế bằng các giá trị...

### 3. Sử dụng package dataprep để làm sạch dữ liệu
