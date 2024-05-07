# BASIC STATISTICS

## 1. Types Of Data
- Categorical
- Numeric
  - Discrete - Dữ liệu rời rạc
  - Continous - Dữ liệu liên tục

## 2. Categorical

### Frequency Distribution Table - Bảng phân phối tần suất

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/d51ba826-e66f-410c-863e-ebfd34a62a8d)

### Bar Chart

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/e20e2b5a-a68c-4eb2-a83c-61a1b0e01cc5)

### Pie Chart

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/5b8e60b1-abfe-4775-81bd-3a7ac14085bf)

### Pareto Diagram

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/674ef092-ae71-49d9-a4fd-e0fd65196350)

## 3. Numeric

### Frequency Distribution Table - Bảng phân phối tần suất

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/87ed0ab4-8a35-4398-846e-039ebe2d1c2c)

### Histogram

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/9c9be1d7-5606-42dd-9a33-e19f7544000e)

### Scatter Plot

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/231d76e6-473a-43b0-a331-4fdf22f6e5a9)

## 4. Measures Of Central Tendency
### Mean
- Ưu điểm
  - Là thước đo phổ biến nhất
  - Trong trường hợp phân phối dữ liệu gần với phân phối chuẩn, trung bình có thể là một ước lượng tốt cho trung tâm của tập dữ liệu.
- Nhược điểm
  - Dễ bị ảnh hưởng bởi ngoại lệ
  - Trong trường hợp phân phối dữ liệu không đối xứng hoặc có skewness, trung bình có thể không phản ánh đúng trung tâm của dữ liệu
### Median
- Ưu điểm
  - Là giá trị nằm chính giữa của dữ liệu
  - Trung vị không bị ảnh hưởng bởi các giá trị ngoại lai trong dữ liệu
  - Phản ánh tốt hơn trong trường hợp phân phối không chuẩn
- Nhược điểm
  - Trung vị không cho biết gì về cách các giá trị phân phối trong dữ liệu.
### Mode
- Ưu điểm
  - Là giá trị phổ biến nhất
  - Mode thường được sử dụng để mô tả dữ liệu phân loại
- Nhược điểm
  - Một tập dữ liệu có thể có nhiều mode
  - Không thích hợp cho dữ liệu liên tục như dữ liệu thời gian hoặc dữ liệu đo lường

## 5. Measure Of Asymmetry
- Skewness
  - Cho biết liệu các quan sát trong tập dữ liệu có tập trung vào 1 phía hay không
  - Lệch phải khi Mean > Median - Các ouliers ở bên phải
  - Lệch trái khi Mean < Median - Các ouliers ở bên trái
  - Đối xứng khi Mean = Median
 
## 6. Measuring How Data Is Spread Out
### Variance
Phương sai là một độ đo của sự biến động của dữ liệu, tức là mức độ mà các điểm dữ liệu phân tán xung quanh giá trị trung bình

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/f69d68d8-9fde-4b5b-8b27-bc19ac6df53d)

### Standard Deviation
Độ lệch chuẩn là căn bậc hai của phương sai và nó đo lường sự biến động của dữ liệu theo đơn vị của dữ liệu.

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/b8b981cb-d4b1-42f7-973e-4452090d4fbf)

### Coefficient of Variation
Hệ số biến thiên là một đại lượng thống kê được sử dụng để đo độ biến động tương đối của một biến so với giá trị trung bình của nó. Cụ thể, CV đo lường sự biến động của một biến so với giá trị trung bình của biến đó theo tỉ lệ phần trăm.

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/c3aa0a40-4999-48fe-b749-7c610343d316)

## 7. Measures of Relationship Between Variables

### Covariance
Hiệp phương sai là một đại lượng thống kê dùng để đo lường mức độ biến thiên cùng chiều giữa hai biến ngẫu nhiên. Nó đo lường sự thay đổi đồng thời của hai biến so với giá trị trung bình của chúng. Nếu covariance dương, điều này ngụ ý rằng khi một biến tăng lên, biến còn lại cũng tăng, và ngược lại. Nếu covariance âm, khi một biến tăng lên, biến còn lại giảm, và ngược lại. Nếu covariance gần bằng 0, không có mối quan hệ tuyến tính giữa hai biến.

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/cb165d70-1f5a-45b4-945b-3e396f05dc19)


### Linear Correlation Coefficient
Hệ số tương quan tuyến tính đo lường mối quan hệ tuyến tính giữa hai biến. Nó được sử dụng để đánh giá độ mạnh và hướng của mối quan hệ tuyến tính giữa hai biến. Pearson correlation coefficient nằm trong khoảng [-1, 1]. Nếu Pearson correlation coefficient gần -1, có một mối quan hệ tuyến tính mạnh âm giữa hai biến. Nếu gần 1, có một mối quan hệ tuyến tính mạnh dương giữa hai biến. Nếu gần 0, không có mối quan hệ tuyến tính giữa hai biến.

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/6f2428ff-895c-4835-9c58-bf2312813bcc)

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/489ab42c-e079-40cd-a0f2-5af7a2cd7271)
