# Data-Pre-processing-and-Analysis

#### Data cleaning (trùng lặp, thiếu, sai, không nhất quán, sai kiểu dữ liệu...) 

#### EDA (phân tích khám phá và tìm ra mối tương quan giữa các biến số và biến phân loại)

#### Chuẩn hoá dữ liệu

#### Feature engineering

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

#### Import các thư viện -> Đọc dữ liệu và xem thông tin cơ bản -> Xác định biến phân loại và biến số (số nguyên hay số thực) -> Kiểm tra dữ liệu có trùng lắp hay không -> Chọn ra các cột dữ liệu có ý nghĩa phân tích -> Chuyển các biến phân loại thành kiểu object (nếu đang ở kiểu str hoặc float) -> Thống kê dữ liệu .describe đối với biến số, .describe(include'object') đối với biến phân loại 

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

## Bài 3: Data Pre-processing (25/5/2024)

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

biến phân loại (dùng nghiệp vụ, chọn ra các thuộc tính có ý nghĩa để phân tích), biến phân loại quan trọng nhất là ý nghĩa thống kê của nó, kqt nó là biến số hay chuỗi (biến phân loại là biến có thể phân nhóm)

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

- xét ngoại lệ trên 1 biến -> vẽ boxplot, histogram

- xét ngoại lệ trên 2 biến -> vẽ scatter plot

- Cách phát hiện ngoại lệ: Thường được sử dụng phổ biến là trực quan hóa (boxplot, histogram, scatterplot), Áp dụng một số quy tắc như z-score ngoài 3 độ lệch chuẩn; tứ phân vị.

- Xử lý ngoại lệ: Hầu hết các cách xử ýl ngoại ệl tương tự như các cách xử ýl với dữ liệu bị thiếu Như xóa mẫu, biến đổi chúng, binning, tạo các riêng biệt, thay thế bằng các giá trị...

### 3. Sử dụng package dataprep để làm sạch dữ liệu

- Sử dụng các hàm trong package dataprep để làm sạch dữ liệu: from dataprep.clean import function_name

function name: có thể là clean_headers), clean_date(), clean_text), clean_df()...

## Bài 4: Exploratory Data Analysis (EDA) (29/5/2024)

### Xác định mối tương quan đối với biến định lượng thì dùng hệ số tương quan

### Xác định mối tương quan đối với biến định tính thì dùng kiểm định giả thuyết

### Thống kê mô tả -> trực quan hoá dữ liệu -> kiểm định giả thuyết 

### 1. Giới thiệu EDA

- EDA thực hiện điều tra ban đầu về dữ liệu để khám phá các mẫu, xu hướng, mối liên quan, phát hiện ngoại lệ, kiếm tra giả thuyết với sự trợ giúp của thống kê tóm tắt và trực quan hóa dữ liệu.

**Mục đích của EDA:**

<img width="1044" alt="Ảnh màn hình 2024-05-29 lúc 18 18 38" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/267bd338-8a88-4602-a9c5-950823708eb6">

**Các phương pháp phân tích EDA**
- Thống kê mô tả (Descriptive Statistics).
- Trực quan hóa dữ liệu (Data Visualization).
- Kiểm định giả thuyết (Hypothesis testing): Chi-squared, ANOVA

**Exploratory Data Analysis (EDA)**

<img width="1143" alt="Ảnh màn hình 2024-06-10 lúc 12 55 56" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/3b3e0a41-0a67-4dc2-a1a7-74e506dae0bc">

### 2. Phân tích một biến, hai biến 

### A. Phân tích một biến

### Phân tích biển phân loại (categorical - biến phân loại)

<img width="906" alt="Ảnh màn hình 2024-05-29 lúc 18 35 24" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/5a2d48d6-9687-47fb-a23d-1bb12a4513f3">

### Phân tích biến liên tục (continuous - biến số liên tục)

- thống kê mô tả -> skew -> vẽ 3 biểu đồ 

<img width="940" alt="Ảnh màn hình 2024-05-29 lúc 18 33 44" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/c2acb78d-d3b7-469f-b411-6701dd54a8ee">

<img width="1061" alt="Ảnh màn hình 2024-06-10 lúc 12 56 24" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/dd9bd297-662d-47d6-92b5-f44d7e942670">

<img width="795" alt="Ảnh màn hình 2024-06-10 lúc 12 59 24" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/da52fd7c-5462-4789-9a28-ac043d57c0b3">

### B. Phân tích hai biến

### Kiểm định cho 2 biến thì có thể dùng T-test

### Kiểm định cho 2 biến trở lên thì dùng ANOVA


- Tìm ra mối quan hệ giữa hai biến: tìm kiếm sự liên kết (association) và không liên kết (disassociation) giữa các biến ở mức ý nghĩa được xác định trước.

**Có thể thực hiện cho bất kỳ sự kết hợp nào của các biến phân loại và liên tục**

Sự kết hợp có thể là: (chapter 4 demo 1)

- Phân loại & Phân loại

<img width="1129" alt="Ảnh màn hình 2024-05-29 lúc 19 16 56" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/23e1d695-9e4b-44de-9fbd-84ef24a979bb">

- Liên tục & Liên tục


- Phân loại & Liên tục

<img width="1063" alt="Ảnh màn hình 2024-05-29 lúc 19 16 09" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/0dcc1650-0c81-4adf-87e3-a29e7eba501d">

**Các loại biểu đồ**

- vẽ sự tương quan giữa 2 biến phân loại: sử dụng biểu đồ heatmap

-  vẽ sự tương quan giữa 2 biến liên tục: biểu đồ scatter

- vẽ sự tương quan giữa 1 biến phân loại và 1 biến liên tục, trục x là biến phân loại, trục y là giá trị trung bình của biến số liên tục: biểu đồ boxplot, biểu đồ bar

### Phân tích hai biến - liên tục và liên tục (biến numberic)

- dùng .corr()

- 0.5 chỉ là số tương đối thôi có khả năng thay đổi dựa trên thực tế 

- kiểm tra nếu trị tuyệt đối >0.5 thì là có mối tương quan, dùng abs để lấy luôn các trường hợp tương quan nghịch

 quan điểm của thầy: 
 
- 0.6 0.7 -> tương quan khá
  
- 0.8 0.9 -> tương quan mạnh
  
- có thể vẽ biểu đồ scatterplot để xem mối tương quan
  
![Ảnh màn hình 2024-06-10 lúc 13 56 32](https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/344450f8-b963-41f6-bb71-a35d0eff92a6)

### Phân tích hai biến - phân loại và phân loại 

<img width="1103" alt="Ảnh màn hình 2024-06-10 lúc 13 58 55" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/127096c3-7107-4de0-9b1d-3bb47e5bee6c">

- có thể vẽ biểu đồ heatmap để xem mối tương quan

- nếu muốn vẽ chồng lên thì thêm stacked=True

### Kiểm định Chi-squared dùng để kiểm định 2 biến phân loại 

**Phát biểu giả thuyết**

- HO: 2 biến phân loại là độc lập

- Ha: 2 biến phân loại là phụ thuộc (có ý nghĩa thống kê, có ý nghĩa đúng cho cả quần thể) (đối thuyết dùng để bác bỏ H0)

**Thu thập dữ liệu**

**Thực hiện kiểm định Chi-squared**

**Kết luận:**

- Nếu p-value<0.05 bác bỏ H0 và chấp nhận Ha.

- Nếu p-value>=0.05 chấp nhận HO.

**Ví dụ**

### Phân tích hai biến - phân loại và liên tục 

### Kiểm định ANOVA (Analysis of Variance) dùng cho 2 biến phân loại và liên tục 

**Kiểm định các biến phân loại (số nhóm >= 2) và biến liên tục**

**Câu hỏi đặt ra ví dụ**

p class (có 3 nhóm 1,2,3)) có ảnh hưởng đến fare hay k 

sex có ảnh hưởng đến fare hay không 

sự kết hợp của pclass và sex có ảnh hưởng đến fare hay không 

## Bài 4: Exploratory Data Analysis (EDA) (tt) (01/6/2024)

### 3. Phát hiện và xử lý outlier

**1. Sử dụng phương pháp IQR**

<img width="528" alt="Ảnh màn hình 2024-06-11 lúc 11 38 19" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/f6c9d963-29ca-463b-873c-055781c63eb0">

**2. Sử dụng phân phối chuẩn, Tìm ngoài khoảng 3 độ lệch chuẩn**

normal distribution chuyển thành standard normal distribution rồi tìm phần +(-) 3 độ lệch chuẩn

<img width="1072" alt="Ảnh màn hình 2024-06-14 lúc 00 44 08" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/efb5089c-622d-4fdf-bfaa-b9100168ce35">

- Định lý thực nghiệm: nếu tệp dữ liệu đạt được phân phối chuẩn như trong hình thì outlier sẽ nằm ngoài khoảng 3 độ lệch chuẩn, khi đó ta có thể loại bỏ outlier

<img width="724" alt="Ảnh màn hình 2024-06-14 lúc 00 49 27" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/a2bbdbe7-a3f1-46d4-87fc-9ba5e4453729">

<img width="744" alt="Ảnh màn hình 2024-06-14 lúc 00 50 41" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/6b8ea07d-077b-4951-9f2e-fca66f6af4b4">

### 4. Các package EDA hữu ích

**dataprep**

**ttth-mds5-analyzer**

- Là gói thư viện hỗ trợ thực hiện các bước phần tích đơn biến và đa biến nhanh chóng

https://pypi.org/project/ttth-mds5-analyzer/

## Bài 5: Feature Engineering (5/6/2024)

### 1. Giới thiệu

- Là quá trình sử dụng kiến thức miền (domain knowledge) về dữ liệu để tạo ra các tính năng giúp thuật toán máy học (Machine Learning algorithms) học được hiệu quả.

**Đặc điểm**

-  tính năng có giá trị đầu vào mới/ rút trích từ những tính năng hiện có của bộ dữ liệu.

-  cô lập và làm nổi bật thông tin chính, giúp thuật toán "tập trung" vào những gì quan trọng

-  Vận dụng domain knowledge để có tính năng thích hợp
  
### 2. Tạo tính năng (feature)

**Xem thông tin các thuộc tính**

- df.info(), df.columns, df.dtypes, df.select_dtypes(),...

**Mã hóa thuộc tính phân loại (Encoding categorical feature)**

- cần mã hoá các biến phân loại dưới dạng định lượng (dạng số) để máy tính có thể đọc và hiểu được

- Có 2 cách mã hoá: label encoder (chuyển thành giá trị số nguyên) và one hot encoder /dummy encoder ( chuyển thành giá trị nhị phân 0 1)

- dùng label encoder khi dữ liệu có thứ tự: ví dụ như xếp loại học lực của học sinh

- dùng one hot encoder /dummy encoder khi dữ liệu không có thứ tự

<img width="1098" alt="Ảnh màn hình 2024-06-15 lúc 02 24 59" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/5c8b46bd-7b0e-405c-b0fd-7ed909c7365b">

**Xử lý các danh mục không phổ biến (uncommon category)**

- ví dụ: gom nhóm các đất nước có số lượng người ít thành other -> dữ liệu nhỏ đưa vô mô hình sẽ bị các dữ liệu lớn áp đảo. vì vậy nếu gom lại vào danh mục chung sẽ ổn 

**Tạo cột nhị phân (Binarizing column)**

- nói về vấn đề phân loại mà chỉ có 2 giá trị => nhị 

**Binning value**

- binning value là những giá trị được phân ra thành khoảng

- đưa biến số thành biến phân loại để xem khoảng mà giá trị rơi vào nhiều nhất

- Các bin được tạo ra bằng cách sử dụng: pd.cut (df[ 'column_name'], bins), bins là số lượng khoảng cách

### 3. Chuẩn hóa dữ liệu (Data Standardization)

- scale (tính lại tỉ lệ để làm giảm bớt sự chênh lệch)

**Log normalization**

- Mục đích: log normalization thường được sử dụng để xử lý các giá trị có sự phân phối lệch hoặc có sự chênh lệch lớn. Nó giúp làm giảm tác động của các giá trị ngoại lệ (outliers) và biến đổi các giá trị sao cho chúng phân phối đều hơn.

- Dùng log để chuẩn hoá 1 biến số khi biến số có phương sai cao/khi đo skew thấy biến số bị lệch phải nhiều/Dữ liệu có các outlier (dùng boxplot để thấy outlier)

- Ví dụ-> sau khi dùng log chúng ta đã thấy đc 2 biến này có mối quan hệ tuyến tính

<img width="1166" alt="Ảnh màn hình 2024-06-19 lúc 21 09 13" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/6227b643-0571-4875-9ccf-64aae778605b">

**Feature Scaling**

- Mục đích: Feature scaling được sử dụng để đảm bảo rằng tất cả các đặc trưng (features) có cùng mức độ quan trọng trong các thuật toán học máy, đặc biệt là các thuật toán nhạy cảm với tỉ lệ như SVM, KNN, và mạng nơ-ron.

- rút tỉ lệ về cho nhỏ và phù hợp hơn, không còn chênh lệch nhiều

- Nếu một tính năng trong bộ dữ liệu có quy mô lớn so với các tính năng khác thì trong các thuật toán được đo bằng khoảng cách Euclide, tính năng có tỷ lệ lớn sẽ trở nên thống trị và cần được chuẩn hóa →dùng Feature scaling giúp cân đối các tính năng.

<img width="723" alt="Ảnh màn hình 2024-06-20 lúc 00 19 20" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/5e5013ba-37be-4ce1-b17f-64d28d6bccbf">

<img width="666" alt="Ảnh màn hình 2024-06-20 lúc 00 19 41" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/c18573b5-060f-4812-a239-fc9b2408e8a6">

**a. Standard Scaler**

- Dùng standard scaler khi chúng ta có phân phối Gaussian

- Sau khi dùng standard scaler thì có phân phối chuẩn tắc (standard normal distribution)

- standard normal distribution có ý nghĩa là trung bình là 0, độ lệch chuẩn là 1

- khi áp dụng standard scaler thì có thể áp dụng cho nhiều cột (truyền nguyên df và standard scaler sẽ tự chọn ra các cột số để chuẩn hoá)

<img width="783" alt="Ảnh màn hình 2024-06-20 lúc 00 28 34" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/990d4103-4ac2-47c7-8d36-df93b3e73262">

<img width="797" alt="Ảnh màn hình 2024-06-20 lúc 00 41 55" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/6543f5b6-bc94-4075-b81b-f9c68a5bed64">

**b. MinMaxScaler**
- Nếu dữ liệu không phải phân phối chuẩn, không có outlier thì dùng min max scaler

**c. RobustScaler**

- Sử dụng khi dữ liệu không phải phân phối chuẩn, có các ngoại lệ trong dữ liệu 

<img width="691" alt="Ảnh màn hình 2024-06-20 lúc 00 42 54" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/533b88ba-ba47-4069-b144-ba2df04512a6">

**d. Binarizer**

- > ngưỡng ánh xạ thành 0, <= ngưỡng thì ánh xạ = 1

- Tự định nghĩa ngưỡng

## Bài 5: Feature Engineering (8/6/2024)

### 4. Chuyển dạng dữ liệu (Data Transformation)

<img width="762" alt="Ảnh màn hình 2024-06-20 lúc 01 24 28" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/c48f2e20-df75-4308-9f97-b203088faa18">

- ví dụ bạn có dữ liệu A nhưng dữ liệu A bị sắp xếp khó cho việc phân tích -> chuyển dạng từ A sang B

**Đặc điểm của tidy data**

- gọn gàng: mỗi biến là 1 cột, mỗi quan sát là 1 hàng

<img width="730" alt="Ảnh màn hình 2024-06-20 lúc 01 29 39" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/ef784b6d-638c-4c60-9058-a780335b5e23">

**a. Chuyển dữ liệu thành Tidy data**

- Vẫn đề về dữ liệu cần khắc phục: Cột chứa giá trị thay vì chứa biến

- Giải pháp: Dùng pd.melt()

<img width="913" alt="Ảnh màn hình 2024-06-20 lúc 01 56 32" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/e8ac50db-ff38-4a40-9080-7149ab3e66f8">

**b. Pivoting data (un-melting data)**

- Biến dòng thành cột

- dùng .pivot()

<img width="872" alt="Ảnh màn hình 2024-06-20 lúc 02 00 25" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/e9f5d076-bd58-40b6-85ce-361017e97695">

**d. Phương thức pivot_table**

- LƯU Ý: trong trường hợp marker và metrics là 2 cột có giá trị lặp lại thì có khả năng báo lỗi khi thực hiện phương thức .pivot

<img width="756" alt="Ảnh màn hình 2024-06-20 lúc 02 06 07" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/1eac555a-919f-45ef-8791-4823dfb81adf">

- Trong trường hợp như vậy thì sử dụng phương thức pivot_table có tham số aggfunc chỉ định cách xử lý trùng lặp giá trị. Ví dụ: Có thể tổng hợp các giá trị trùng lặp bằng cách lấy trung bình cộng (np.mean)

<img width="658" alt="Ảnh màn hình 2024-06-20 lúc 02 09 23" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/6cba2ceb-cec8-42d6-a330-ab7289808523">

## Bài 6: Natural Language Processing (8/6/2024)

### 1. Tổng quan

**Natural Language Processing**

**Một số nhiệm vụ của NLP**

**Một số ứng dụng NLP thông dụng**

**Một số quy trình NLP cơ bản**

### 2. Các thư viện hỗ trợ NLP phổ biến

### 3. Text data Pre-processing

**Chuẩn hóa text**

### 4. Text data Transformation 

### 5. Tiền xử lý tài liệu tiếng Việt

- Tokenizer: Là một công cụ từ tensorflow.keras dùng để token hóa văn bản, tức là chuyển đổi các từ trong văn bản thành các số nguyên (token IDs) để mô hình máy học có thể xử lý.

Ví dụ:

Bình luận: "tôi rất tốt" → [1, 2, 3] (dựa trên từ điển giả định ở trên).

Bình luận: "tôi tốt" → [1, 3].

- pad_sequences: Là hàm dùng để chuẩn hóa độ dài của các chuỗi số (sequences) đã được token hóa, đảm bảo tất cả các chuỗi có cùng độ dài bằng cách thêm đệm (padding) hoặc cắt ngắn (truncation).

Chuỗi: ["I", "love", "NLP"] (dài 3 token).

Padding đến độ dài 5: ["I", "love", "NLP", "[PAD]", "[PAD]"].

Chuỗi: ["I", "love", "to", "learn", "NLP", "and", "AI"] (dài 7 token).

Truncation đến độ dài 5: ["I", "love", "to", "learn", "NLP"].

#### Thư viện underthesea.word_tokenize
- word_tokenize từ thư viện underthesea là công cụ tách từ (word segmentation) chuyên biệt cho tiếng Việt, nhận diện các cụm từ ghép như "tẩy trang", "sản phẩm".

## Bài 7: Linear Regression (9/6/2024)

- ref: nguyễn vân tuấn, lê thị kim ánh, đặng thế khoa, khuất thị 

- Linear regression là thuật toán thuộc nhóm Supervised Learning sử dụng cho regression

- biến dự báo dự đoán là biến số liên tục

- Là một trong những chủ đề đầu tiên cần biết khi học về mô hình tiên đoán (learning predictive modeling).

### 1. Hồi quy tuyển tính (Linear Regression)

#### Ý tưởng: 
- Tmì mối quan hệ giữa numerical output và input variables. Dependent variable liên tục, còn các Independent variable có thể liên tục hoặc rời
rạc.

**Mô hình hóa mối quan hệ dưới dạng tuyến tính**

<img width="1033" alt="Ảnh màn hình 2024-06-09 lúc 08 45 02" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/160599d1-b1eb-478a-a462-39b36834d30f">

**Xây dựng hàm mất mát**

<img width="1057" alt="Ảnh màn hình 2024-06-09 lúc 08 57 44" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/7da73513-ed60-4edf-9838-99991814a824">

- mục đích: xác định các hệ số beta để tìm ra giá trị nhỏ nhất của hàm mất mát

<img width="1050" alt="Ảnh màn hình 2024-06-09 lúc 09 00 52" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/a82f035f-d988-4cd9-a209-ca1290fa4477">

- trên trục x là input, trên trục y là output

**Mô hình**

đường thẳng màu đỏ: y = ax + b 

- lưu ý: mô hình hồi quy tuyến tình rất kỵ nhiễu (outlier)

![Ảnh màn hình 2024-06-09 lúc 09 06 34 (2)](https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/70e6682f-8bcf-452b-a539-267b69ae0dad)

**Least Squares Algorithm**

<img width="1035" alt="Ảnh màn hình 2024-06-09 lúc 09 11 16" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/30bcd856-beba-4d3f-8387-0de5770e0ac6">

- sai số là y - y thực tế = e

- số dư là (y - y thực tế)^2

<img width="760" alt="Ảnh màn hình 2024-06-09 lúc 09 13 18" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/72457011-b3da-45be-99c5-172fca775687">

#### Phân loại: Simple Linear Regression và Multiple Linear Regression

<img width="724" alt="Ảnh màn hình 2024-06-09 lúc 09 16 33" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/f48075e2-538a-4666-a22a-bb5b2b0a869b">

### 2. Hồi quy đơn biến (Simple Linear Regression)

- Simple Linear Regression là phương pháp tìm hiểu mối quan hệ giữa hai biến: Biến độc lập (predictor/independent variable): X và Biến phụ thuộc (response/dependent variable) là biến mà chúng ta muốn dự đoán: Y

- tìm hiểu biến x và biến y có tương quan với nhau không

- trong trường hợp ở đây là có biến phân loại thì 1 biến phân loại sẽ sử dụng kiểm định giả thuyết chi2 và 2 biến phân loại trở lên thì sử dụng ANOVA

**Function**

<img width="999" alt="Ảnh màn hình 2024-06-09 lúc 09 22 15" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/df1007cb-17e5-403a-bf0b-9ddcc355913d">

### 3. Hồi quy đa biến (Multiple Linear Regression)

- Nếu chúng ta muốn sử dụng nhiều biến hơn trong mô hình (X1, X2,...) để dự đoán (Y), chúng ta có thể sử dụng Multiple Linear Regression.

- Multiple Linear Regression rất giống với Simple Linear Regression, nhưng phương pháp này được sử dụng để giải thích mối quan hệ giữa một biến phụ thuộc và hai hoặc nhiều biến độc lập.

- Hầu hết các mô hình hồi quy (regression model) trong thực tế liên quan đến nhiều yếu tố dự đoán.
  
**Fuction**

<img width="950" alt="Ảnh màn hình 2024-06-09 lúc 09 25 36" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/4bc3e683-8724-4181-b2b2-cf04ae32d5bb">

### 4. Hồi quy đa thức (Polynomial Regression) 

### 5. Một số kỹ thuật thực hiện trên mô hình

## Bài 8: Logistic Regression (15/06/2024)

### 1. Logistic Regression

**Phương trình toán học (Sigmoid)**

<img width="823" alt="Ảnh màn hình 2024-06-15 lúc 09 04 43" src="https://github.com/berylhoang2501/Data-Pre-processing-and-Analysis/assets/152646327/a401e321-dc8b-4487-be07-f6120da19052">


### 2. Xây dựng Logistic Regression Model

## Bài 9: Handling Imbalanced Dataset (16/06/2024)

### Giới thiệu

### Chiến thuật làm việc với dữ liệu mất cân bằng (Imbalanced Dataset)
