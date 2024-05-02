# Eigenvectors and Eigenvalues

## 1. Determinant
Định thức là một khái niệm trong toán học và đại số tuyến tính, được sử dụng để đo lường tính "đặc biệt" của một hệ thống các phần tử hoặc mối quan hệ giữa chúng. Trong đại số tuyến tính, định thức của một ma trận là một số thực hoặc số phức được tính từ các phần tử của ma trận theo một cách định sẵn.

``` bash
X = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])
np.linalg.det(X)
```

## 2. Eigenvectors and Eigenvalues

``` bash
lambdas, v = np.linalg.eig(X)
```


# 3. Eigenvalues and Determinant Relationship
Định thức của 1 ma trận là tích của tất cả trị riêng của ma trận đó
