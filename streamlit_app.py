import streamlit as st
import pandas as pd



df = pd.DataFrame(
    [
        {"Member Name": "Myles Ezeanii", "Proposal Contribution": "Results and Discussion"},
        {"Member Name": "Amir Salah", "Proposal Contribution": "Github & Streamlit"},
        {"Member Name": "Dennis Hantman", "Proposal Contribution": "Video"},
        {"Member Name": "Caleb Asress", "Proposal Contribution": "Intro/Background"},
        {"Member Name": "Alexandre Chaker", "Proposal Contribution": "Methods"},
    ]
)
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
To reach our goal of a machine learning solution that can help consumers decide on cars, we implemented a neural network model based on the Car Evaluation data set. The features included in this data set are buying price, maintenance price, number of doors, seating capacity, luggage boot size, and estimated car safety.\n
We took many preprocessing steps before analyzing the data. First, we split the dataset into training and test sets. We do this in order to gauge how well the model is learning the training data as it goes along. Then, we use mean imputation to replace any missing data points in our set with the mean. This allows us to have a more complete set to work with, and increases our model accuracy. Then, we separate the categorical features, such as car safety, from the numerical features, such as number of doors. We then create transformers for our categories, using the mean of values for our numerical features and One-Hot encoding on our categorical features.
Finally, we fit the preprocessor on the training data.\n
Now that the data is preprocessed, we start supervised learning on the data. Our model consists of an input layer with 6 features, a hidden layer with 64 units and ReLU activation, and an output layer with 4 units and softmax activation. Then, we fit a LabelEncoder on the target variable and transform it. This allows our model to numerically recognize our variables. Finally, we train our model on the variable using 10 epochs and a batch size of 32.
\n
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
""")
st.header("Video Presentation:\n")
st.write("""
https://youtu.be/YUN-Sh4Gahs?si=C9UTUkMHF5Deerqu\n""")
st.header("Contribution Table:")
st.dataframe(df, use_container_width=True)
st.header("Gantt Chart:")
st.write("https://docs.google.com/spreadsheets/d/1gD6TI02N_67U_YAIVeduaDORvfWMKicr/edit?gid=2146609855#gid=2146609855\n")

st.header("Works Cited")
st.write("""
[1] I. Bauer, L. Zavolokina, and G. Schwabe, “Is there a market for trusted car data?,” Electronic Markets, vol. 30, no. 2, pp. 211–225, Sep. 2019, doi: https://doi.org/10.1007/s12525-019-00368-5.\n
[2] A. Das Mou, P. K. Saha, S. A. Nisher, and A. Saha, “A Comprehensive Study of Machine Learning algorithms for Predicting Car Purchase Based on Customers Demands,” 2021 International Conference on Information and Communication Technology for Sustainable Development (ICICT4SD), Feb. 2021, doi: https://doi.org/10.1109/icict4sd50815.2021.9396868.\n
[3] S. Singh, S. Das, A. Sajwan, I. Singh, and A. Alok, “Car Recommendation System Using Customer Reviews,” *International Research Journal of Engineering and Technology (IRJET)*, vol. 9, no. 10, pp. 983–989, Oct. 2022
"""
)

