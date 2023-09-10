# Nonlinear-regression-Gaussian-and-Kernel-Density-estimation
Nonlinear regression, Gaussian and Kernel Density estimation visualization, Density estimation and performance evaluation.


# NONLINEAR REGRESSIONS AND DENSITY ESTIMATIONS
## OBJECTIVE
1.	To understand the difference of between linear and polynomial regression
2.	To apply density estimation using Gaussian and Kernel Density Estimation
## PROCEDURE
### Preparation
First, we import the data (Sklearn.iris) we will use, and the libraries such as numpy, matplotlib. After that we split X and y for 80% X values for training and 20% y values for testing. There must be four data which are named training and testing for X, training, and testing for y. 
We used training data for train to data, also we used the data we trained for testing.
## Q1. Nonlinear Solution
### Linear solution 
We calculate the coefficient and intercept by using sklearn linear model. Our coefficient is 349.20109567 and Intercept is 151.7825221509722. MSE 5919.33, Cost is -0.15.

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/b35ee2af-9c47-45ac-9533-55db246b3ef2)
### Linear Regression Graph

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/6feb2b4c-efec-4925-bf47-7c6814f6d3b8)

### Polynomial Regression Solution.
We calculate the solution by using sklearn.preprocessing PolynomialFeatures. Our coefficient is changing respect to data, and Intercept is 153.70361. MSE has reduced to 5790.67 and cost has reduced -0.13. Best degree for us is 7. As seen in the calculation after 7 degree, the MSE is increasing.
### Degree 2
 ![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/088c1500-3b42-4141-af7c-c7839c27f56a)

 ![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/a0214f72-599a-4f54-9c4d-d983803ded8b)

### Degree 3
 
 ![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/230f6570-3eba-49ff-b9df-a95f09d2499c)
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/19caa2cb-4e43-4dae-a873-853156fef82d)

### Degree 7
  ![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/e67e4866-6b51-48fd-8641-7cbf50effdc9)
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/3cf085b5-ee97-4aef-ae23-a3a5cb959923)

### Degree 8
 
 ![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/885861b8-20eb-4798-a300-c1be28231fb8)

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/74d690b3-cf9f-4a2a-a9c7-2e37e258ca51)

### Gaussian Solution
We calculate the solution by using sklearn.gaussian_process GaussianProcessRegressor.
When we used Gaussian Kernel Regression, the MSE increased 5793.19 and Cost is remained the same with Polynomial Solution even if degrees are changed. Best degree for gaussian solution is 2.

Degree 2
 ![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/86415b26-7bbd-44cd-bdf0-d115927c25af)
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/a63784a9-0c44-4089-8009-167d11f95f51)

 

Degree 3
 ![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/c5e855fb-731c-4b33-ba71-738130bbee83)
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/468b9541-e67a-4107-9f09-33b97c420ba9)

 
## Q2. Gaussian Mixture and Kernel Density
### Gaussian Mixture
For this question, we used sepalwidth (1) and petal length (2) features. To classificate the model, we used gaussian mixture model.
Our classification graphs for different components.
### Component 1
Adjusted Rand Score: 0

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/f26fb4a2-1649-4b4d-a718-e4f5841af9bc)
### Component 2.
Adjusted Rand Score: 0.5611988052626607

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/6c433a37-e614-43cc-91d7-90cef2107c61)
### Component 3.

Adjusted Rand Score: 0.8572208267220754

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/d47bc49c-92d5-41d6-8d54-3042ff411e05)
### Component 4.

Adjusted Rand Score: 0.7434978631440832

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/d68e7d01-4d3b-46b2-9942-f9dfb462a586)
### Component 5.

Adjusted Rand Score: 0.6764457303913506

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/1259f25f-dd48-4de5-865e-23e3d77e9d52)
## Classification Performance
The best component for classification is 3 which value is 0.8572208267220754. Also, our data accuracy score is 0.97.
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/eb8197f5-dc7e-4072-b57e-cd5a0d7a0835)
### Kernel Density Estimation
For this question, we used sepalwidth (1) and petal length (2) features. To classificate the model, we used sklearn kernel model.
Our density graphs for different bandwidth.


### Bandwidth 0.5

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/f20c3245-0592-4134-8539-e092b1c4d9d4)
### Bandwidth 0.7
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/f9331414-2ca4-4aac-a37e-03d8bf41ae01)

### Bandwidth 1
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/d88e9744-fe22-42e2-83ac-9532825c0a14)

## Q3. Use All features
When we used all features, the best component is 3 with a value 0.90387. According to Naïve bayes, the accuracy score is 0.96 and there are 6 mislabeled out of a total 150 points.
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/4a654595-d3bf-4c98-82e7-a6f9bfb8c350)
### Bandwidth 0.6

![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/5945f930-ce7b-4a33-95e9-3fb553c2dc1d)


### Bandwidth 0.7
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/ab8cae1e-0f1c-4c50-99f6-bea1eafb5394)

### Bandwidth 1
![image](https://github.com/onlinEngineer/Nonlinear-regression-Gaussian-and-Kernel-Density-estimation/assets/70773825/026c8654-b114-4edb-b6b5-df445003d1b4)

# CONCLUSION
In this assignment, we leant what is nonlinear regression, difference linear and nonlinear regression, what is classification, how we classify the given model etc.
In nonlinear regression, we use polynomic equation. When we increase the degree of polynomial, the model is more fitting the data and MSE is reducing. But If we increase the degree of model, we encounter with the overfitting problem. In first question, the MSE is increasing after seventh degree, so the seventh best degree choice for us.
For second question, we tried different number of component and the best component was 3 for us. After third component, the rand score starts to reduce.
The kernel density estimation, we used two different kernel which are gaussian and tophat. Both estimations gives us different solution for each bandwidth. While gaussian is smooth and smaller, the tophat was bigger and not smooth. 
