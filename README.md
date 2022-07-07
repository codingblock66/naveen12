# single-page-portfolio-website-
design a single page portfolio website using React and any CSS library
# Diabetes-Prediction-Using-Machine-Learning


Diabetes is an illness caused because of high glucose level in a human body. Diabetes should not be ignored if it is untreated then Diabetes may cause some major issues in a person like: heart related problems, kidney problem, blood pressure, eye damage and it can also affects other organs of human body. Diabetes can be controlled if it is predicted earlier. To achieve this goal this project work we will do early prediction of Diabetes in a human body or a patient for a higher accuracy through applying, Various Machine Learning Techniques. Machine learning techniques Provide better result for prediction by con- structing models from datasets collected from patients. In this work we will use Machine Learning Classification and ensemble techniques on a dataset to predict diabetes. Which are K-Nearest Neighbor (KNN), Logistic Regression (LR). The accuracy of model is noted . The Project work gives the accurate or higher accuracy model shows that the model is capable of predicting diabetes effectively. 

![image](https://user-images.githubusercontent.com/103196322/164281832-82b55887-6482-4dfe-b412-8f2ad8bc3cb5.png)


Dataset Description- the data is gathered from UCI repository which is named as Pima Indian Diabetes Da- taset. The dataset have many attributes of 768 patients.

Data Preprocessing- 

Data preprocessing is most im- portant process. Mostly healthcare related data contains missing vale and other impurities that can cause effective- ness of data. To improve quality and effectiveness obtained after mining process, Data preprocessing is done. To use Machine Learning Techniques on the dataset effectively ths process is essential for accurate result and successful prediction. For Pima Indian diabetes dataset we need to perform pre processing in two steps.

Missing Values removal- Remove all the instances that have zero (0) as worth. Having zero as worth is not possi- ble. Therefore this instance is eliminated. Through elimi- nating irrelevant features/instances we make feature subset and this process is called features subset selection, which reduces diamentonality of data and help to work faster.

Splitting of data- After cleaning the data, data is nor- malized in training and testing the model. When data is spitted then we train algorithm on the training data set and keep test data set aside. This training process will produce the training model based on logic and algorithms and val- ues of the feature in training data. Basically aim of normal- ization is to bring all the attributes under same scale.

Apply Machine Learning- When data has been ready we apply Machine Learning Technique. We use different classification and ensemble techniques, to predict diabetes. The methods applied on Pima Indians diabetes dataset. Main objective to apply Machine Learning Techniques to analyze the performance of these methods and find accura- cy of them, and also been able to figure out the responsi- ble/important feature which play a major role in prediction.

Logistic Regression- Logistic regression is also a su- pervised learning classification algorithm. It is used to es- timate the probability of a binary response based on one or more predictors. They can be continuous or discrete. Lo- gistic regression used when we want to classify or distin- guish some data items into categories.

It classify the data in binary form means only in 0 and 1 which refer case to classify patient that is positive or nega- tive for diabetes.

MODEL BUILDING

This is most important phase which includes model build- ing for prediction of diabetes. In this we have implemented various machine learning algorithms which are discussed above for diabetes prediction.

Procedure of Proposed Methodology-

Step1: Import required libraries, Import diabetes dataset.
![Screenshot (61)](https://user-images.githubusercontent.com/106168483/177718080-746c1563-4c6b-469a-9c18-fb96e2263108.png)


Step2: Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees..
![Screenshot (62)](https://user-images.githubusercontent.com/106168483/177718263-3e0ea41b-b3f5-46db-b0ed-01369f9a2392.png)

Step3: A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.
![Screenshot (63)](https://user-images.githubusercontent.com/106168483/177718467-35e9150b-d8e4-4b3a-bf4c-0a52fe7e5ba8.png)


Step4: Select the machine learning algorithm i.e. K- Nearest Neighbor, Support Vector Machine, Decision Tree, Logistic regression, Random Forest and Gradient boosting algorithm.The k-nearest neighbors (KNN) algorithm is a data classification method for estimating the likelihood that a data point will become a member of one group or another based on what group the data points nearest to it belong to. The k-nearest neighbor algorithm is a type of supervised machine learning algorithm used to solve classification and regression problems. 
![Screenshot (64)](https://user-images.githubusercontent.com/106168483/177718506-96dc1c29-6275-4e80-ae18-e90ba1bd05a7.png)


Step5: Linear regression models are used to identify the relationship between a continuous dependent variable and one or more independent variables. When there is only one independent variable and one dependent variable, it is known as simple linear regression, but as the number of independent variables increases, it is referred to as multiple linear regression. For each type of linear regression, it seeks to plot a line of best fit through a set of data points, which is typically calculated using the least squares method..
![Screenshot (65)](https://user-images.githubusercontent.com/106168483/177719027-902e4f1e-e43e-40fd-99a8-9a11add4e3d2.png)


Step6: Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter.
![Screenshot (66)](https://user-images.githubusercontent.com/106168483/177719088-83266858-b6fe-42d1-a1d1-3b88c4b9f38a.png)


Step7:Matplotlib is a python library used to create 2D graphs and plots by using python scripts. It has a module named pyplot which makes things easy for plotting by providing feature to control line styles, font properties, formatting axes etc.
![Screenshot (67)](https://user-images.githubusercontent.com/106168483/177719165-63963672-24e0-446b-85ae-d197b8911dba.png)


Step8: After analyzing based on various measures con- clude the best performing algorithm.

![Screenshot (69)](https://user-images.githubusercontent.com/106168483/177719316-b6e4a318-59db-4f0b-81a2-6bd65d54e1cd.png)

The ROC curve is a fundamental tool for diagnostic test evaluation. The diagnostic performance of a test, or the accuracy of a test to discriminate diseased cases from normal cases is evaluated using Receiver Operating Characteristic (ROC) curve analysis
![Screenshot (70)](https://user-images.githubusercontent.com/106168483/177719378-36a4379e-da82-43e4-942c-7dbc643e8979.png)




REFERENCES

Debadri Dutta, Debpriyo Paul, Parthajeet Ghosh, "Analyzing Feature Importances for Diabetes Prediction using Machine Learning". IEEE, pp 942-928, 2018.

Tejas N. Joshi, Prof. Pramila M. Chawan, "Diabetes Prediction Using Machine Learning Techniques".Int. Journal of Engineer- ing Research and Application, Vol. 8, Issue 1, (Part -II) Janu- ary 2018, pp.-09-13
