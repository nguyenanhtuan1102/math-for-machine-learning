# INTRODUCE FOR LINEAR ALGEBRA

## 1. Data Structures For Linear Algebra

### Tensor

### Scalar

Là một số đơn lẻ và không có chiều
  
``` bash
# Scalar
scalar_number = tourch.tensor(25)
```

### Vector

Là mảng một chiều của các số

``` bash
# Vector
vector1 = numpy.array([1,2,3])

# Vector Matrix
vector2 = numpy.array([[1,2,3]])

# Vector Transposition (0,3) to (3,0)
vector_t1 = vector1.T

# Vector Transposition (1,3) to (3,1)
vector_t2 = vector2.T

# Vector in Pytorch and Tensorflow
vector = tourch.tensor([1,2,3])
```

### Norms

- Norms là một lớp hàm cho phép ta định lượng độ dài của 1 vector cho trước
- L2 Norm hay còn gọi là Khoảng cách Euclid là phương pháp quan trọng và phổ biến nhất
- Unit Vector là một vector có độ dài bằng 1

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/0aacaf0c-ad32-4fe5-999e-e34481bee8e7)

``` bash
# Calculate L2 Norm
x = np.array([25, 2, 5])
np.linalg.norm(x)
```

### Matrix
- Ma trận là một mảng 2 chiều của các số
- Viết dưới dạng (m * n)

``` bash
# Create a matrix (3,2)
X = np.array([[1,2],[4,5],[7,8]])

# Create a matrix in Pytourch and Tensorflow
X = tourch.tensor([[1,2],[4,5],[7,8]])

# Create a zeros matrix image in Pytorch with 28*28 size, 32 images, 3 chanels
image_pt = tourch.zeros([32,28,28,3])
```

## 2. Tensor Operations
### Tensor Transposition
- Chuyển vị của Scalar là chính nó
- Chuyển vị của Vector và Matrix là chuyển đổi giữa hàng và cột

``` bash
# Create a matrix (3,2)
X = np.array([[1,2],[4,5],[7,8]])

# Matrix Transposition
X.T
```
### Tensor Calculation
Tích của Tensor với một số bằng tích của số đó với tất cả các số trong Tensor

``` bash
# Tích Tensor với 1 số
X * 2
```

Tổng của Tensor với 1 số bằng tổng của số đó với tất cả các số trong Tensor
``` bash
# Tổng Tensor với 1 số
X + 2
```

Tổng của 2 Tensor cùng chiều là tổng của từng phần tử trong Tensor này với Tensor còn lại
``` bash
# Tổng Tensor với Tensor
X + Y
```

Tổng của tất cả các phần tử trong Tensor (min, max, mean)
``` bash
# Tổng của tất cả các phần tử trong Tensor - Numpy
X.sum()

# Tổng của tất cả các phần tử trong Tensor - Pytorch
tourch.sum(X)
```

Tổng của tất cả các phần tử trong Tensor trên 1 hàng và 1 cột (min, max, mean)
``` bash
# Tổng của tất cả các phần tử trong Tensor trên 1 hàng - Numpy
X.sum(axis=0)

# Tổng của tất cả các phần tử trong Tensor trên 1 cột- Numpy
X.sum(axis=1)

# Tổng của tất cả các phần tử trong Tensor trên 1 hàng - Pytorch
tourch.sum(X, 0)

# Tổng của tất cả các phần tử trong Tensor trên 1 cột- Pytorch
tourch.sum(X, 1)
```

### Hadamard Product
Là tích của 2 Matrix cùng chiều bằng cách nhân từng phần tử của ma trận này tương ứng với từng phần tử của ma trận kia

``` bash
# Hadamard product
X * Y
```

### Dot Product
Tích vô hướng của 2 Vector là tích của Vector A với Vector B
``` bash
# Dot product
np.dot(X,Y)
```

### Substitution and Elimination

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/a978d4e2-e4d1-4515-92c0-ec9cbda90844)

## 3. Matrix Propeties

### Frobenius Norm

Bằng căn bậc 2 của tổng tất cả các phần tử bình phương trong ma trận

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/eafb5f19-3d9d-4b21-8013-4beb9a8b92f8)

``` bash
# Calculate Frobenius Norm
x = np.array([25, 2, 5],[1,2,4],[4,12,11])
np.linalg.norm(x)
```

### Matrix Multiplication

Matrix với Vector

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/1ff2c3ce-c8c6-4ded-99a0-002c2331e504)

``` bash
# Matrix với Vector
np.dot(A,B)
```

Matrix với Matrix

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/a61cf4fb-a4b0-4bcd-8604-18b6a3d95369)

``` bash
# Matrix với Matrix
np.dot(A,B)
```

### Symmetric Matrix

Ma trận đối xứng là ma trận mà chuyển vị của nó bằng chính nó

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/0f9b1de2-1c6a-4c04-a002-411d273d7a29)

### Identity Matrix

Ma trận đơn vị là một loại ma trận vuông trong đó tất cả các phần tử trên đường chéo chính (từ góc trên bên trái đến góc dưới bên phải) đều có giá trị bằng 1, và tất cả các phần tử còn lại đều bằng 0.

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/71dc6121-594d-4cf8-bb17-bd2aef510165)

### Matrix Inversion

Ma trận nghịch đảo của một ma trận vuông 𝐴 là một ma trận sao cho tích của nó với ma trận ban đầu là ma trận đơn vị và tích của ma trận ban đầu với ma trận nghịch đảo cũng là ma trận đơn vị.

``` bash
# Ma trận nghịch đảo
X_inv = np.linalg.inv(X)
```

### Diagonal Matrix

Ma trận đường chéo là một loại ma trận trong đó tất cả các phần tử nằm ngoài đường chéo chínhđều bằng 0. Các phần tử trên đường chéo chính có thể có giá trị khác nhau.

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/49a594b5-1492-489a-95dc-8e5cd2e7d205)

### Orthofonal Matrix

Ma trận trực giao là một ma trận vuông mà tích của nó với chuyển vị của nó là ma trận đơn vị.

![image](https://github.com/tuanng1102/math-for-machine-learning/assets/147653892/5f3965bb-c980-4477-ae5c-e0da5a72b973)
