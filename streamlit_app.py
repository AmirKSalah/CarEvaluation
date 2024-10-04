import streamlit as st

st.title("Car Evaluation")
st.write("""
Amir Salah, Myles Ezeanii, Dennis Hantman, Caleb Asress, Alexandre Chaker \n
Max Mahdi Roozbahani \n
CS4641/CS7641\n
04 October 2024\n
""")
st.header("Introduction/Background")
st.write("""
Deciding on what car to buy can be a tough decision, especially for people who don’t know a lot about them. We wanted to create something that would help people in this situation who may not have access to a ton of resources to help make the right decision for them. Thus, our group decided on making a car recommendation system that recommends vehicles based on their features. In one study regarding a similar idea, Singh et al. proposed a car recommendation system that utilized NLP and machine learning techniques to analyze both user reviews and user preference to ultimately recommend a car or not. Our project aims for something similar with more of a focus on the features of the car.\n
Our dataset contains information on price, size, safety, tech, and more features for over 1000 different cars. We thought this would be a good dataset to start off with as there are multiple features we can use to judge the car on.\n
Dataset Link:  https://archive.ics.uci.edu/dataset/19/car+evaluation\n
         """)
st.header("Methods")
st.write("""
Data Preprocessing:
• One-Hot or Label Encoding to convert category variables to numerical values.\n
• Manage missing values, even if the dataset appears complete.\n
• Scaling features for algorithms that require them, such as Support Vector Machine (SVM) and k-Nearest Neighbors (k-NN).\n
• A train-test split to divide data for model evaluation.\n
Machine Learning Algorithms:\n
• Supervised learning:\n
• Decision Trees for categorical data and straightforward to interpret.\n
• Random Forest to enhance accuracy by minimizing overfitting.\n
• Support Vector Machines (SVM) for classification jobs and to handle high-dimensional data.\n
Scikit-learn:\n
• OneHotEncoder() or LabelEncoder() for encoding.\n
• train_test_split() to split data.\n
• DecisionTreeClassifier() for decision trees.\n
• RandomForestClassifier() for random forest.\n
• SVC() to support vector machines.\n
• K Means() for clustering.\n
• StandardScaler() to scale.\n
         """)

st.header("(Potential) Results and Discussion:\n")
st.write("""
For our project, we will be using four separate quantitative metrics to determine the efficiency of the model. To start, we will use accuracy to help us find its overall correctness. Because it measures the proportion of cars correctly classified as either acceptable or unacceptable, the accuracy will be equal to ​ (True Positives + True Negatives)/(Total Number of Predictors).  Using this, we expect to get a good sense of how many correct predictions are given across our dataset.\n
Additionally, we will use a confusion matrix to break our results down to true and false positives/negatives. This will allow us to see where our model is making mistakes and determine if these false positives / negatives are costly to our model. Thirdly, we intend to use precision and recall to further determine the consequences and how significant they are. For precision, which checks how many cars predicted as acceptable are actually acceptable, our formula will be  (True Positives) / (True Positives + True Negatives). For recall (sensitivity) which measures how many of the actual acceptable cars were correctly classified, our formula will be  (True Positives)/(True Positives + False Negatives). To find the harmonic mean between the two, we will find the F1-score given in the form of 
2*(Precision*Recall)/(Precision + Recall). This is useful for finding a balance between precision and recall and can help us if we start seeing imbalances in the class prediction. \n
Finally, we will use Receiver Operating Characteristic - Area Under the Curve (ROC-AUC) to see if the dataset is imbalanced. Our ROC-AUC will be equal to the area under the curve plotting the True Positive Rate.\n
For our project, our goal is to create a model which can accurately determine the acceptability of a car based on the standards we provide. This will allow for users to decide accurately whether or not the cars they intend to use are adequate and we expect the results to provide us just that.\n
 Video Presentation:\n


Works Cited
S. Singh, S. Das, A. Sajwan, I. Singh, and A. Alok, “Car Recommendation System Using Customer Reviews,” *International Research Journal of Engineering and Technology (IRJET)*, vol. 9, no. 10, pp. 983–989, Oct. 2022
"""
)
