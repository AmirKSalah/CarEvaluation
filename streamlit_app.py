import streamlit as st
import pandas as pd



df = pd.DataFrame(
    [
        {"Member Name": "Myles Ezeanii", "Proposal Contribution": "Model/Notebook Creation"},
        {"Member Name": "Amir Salah", "Proposal Contribution": "Streamlit & Methods"},
        {"Member Name": "Dennis Hantman", "Proposal Contribution": "Intro/Background"},
        {"Member Name": "Caleb Asress", "Proposal Contribution": "Did Not Participate"},
        {"Member Name": "Alexandre Chaker", "Proposal Contribution": "Problem Definition"},
    ]
)
st.title("Car Evaluation")
st.write("""
Amir Salah, Myles Ezeanii, Dennis Hantman, Caleb Asress, Alexandre Chaker \n
Max Mahdi Roozbahani \n
CS4641/CS7641\n
11 November 2024\n
""")
st.header("Introduction/Background")
st.write("""
Many people may not have resources available that could help them when purchasing a new car. To help such people, our group decided to make a car recommendation system that could help users pick a car based on desired features. In one study we found regarding a similar idea, Singh et al. proposed a car recommendation system that utilized different machine learning techniques to analyze both user reviews and user preferences to ultimately recommend a car or not. In our project, we aim to develop a similar system with more of a focus on the features of the car, utilizing supervised learning methods to make recommendations.\n
The dataset we found contains information on price, size, safety, technology, and more for over 1000 different cars. Since there is a large variety of features we can use to recommend cars, we thought this would be a good dataset to use. Applying supervised learning strategies to this dataset ultimately allows us to give accurate recommendations based on users’ desired features.\n
Dataset Link:  https://archive.ics.uci.edu/dataset/19/car+evaluation\n
         """)
st.header("Problem Definition")
st.write("""
The process of selecting the proper car can be difficult, particularly for people who lack considerable automotive knowledge or access to adequate resources. Our project tries to solve this problem by creating an automobile recommendation system using machine learning techniques. The major goal is to develop a model capable of categorizing cars as "unacceptable," "acceptable," "good," and "very good" based on their qualities. This will allow consumers to make informed selections based on their preferences and needs.\n
This project’s dataset consists of a variety of car parameters from the UCI Car Evaluation dataset, including the purchase price, maintenance cost, number of doors, seating capacity, luggage boot size, and safety ratings. The dataset is well-suited to this task since it includes a wide variety of features that can influence car acceptability. By emphasizing these characteristics, our project aims to create an effective tool that streamlines the car purchasing process for prospective purchasers.\n
To achieve this, we used supervised learning models, which are intended to examine and forecast the acceptability of a car based on the attributes presented. The models were chosen because they are good at handling categorical and numerical data, resulting in robust and dependable suggestions.\n


""")

st.header("Methods")
st.write("""
To reach our goal of a machine learning solution that can help consumers decide on cars, we implemented a neural network model based on the Car Evaluation data set. The features included in this data set are buying price, maintenance price, number of doors, seating capacity, luggage boot size, and estimated car safety.\n
We took many preprocessing steps before analyzing the data. First, we split the dataset into training and test sets. We do this in order to gauge how well the model is learning the training data as it goes along. Then, we use mean imputation to replace any missing data points in our set with the mean. This allows us to have a more complete set to work with, and increases our model accuracy. Then, we separate the categorical features, such as car safety, from the numerical features, such as number of doors. We then create transformers for our categories, using the mean of values for our numerical features and One-Hot encoding on our categorical features.
Finally, we fit the preprocessor on the training data.\n
Now that the data is preprocessed, we start supervised learning on the data. Our model consists of an input layer with 6 features, a hidden layer with 64 units and ReLU activation, and an output layer with 4 units and softmax activation. Then, we fit a LabelEncoder on the target variable and transform it. This allows our model to numerically recognize our variables. Finally, we train our model on the variable using 10 epochs and a batch size of 32.
\n
Scikit-learn Tools Used:\n
• OneHotEncoder() or LabelEncoder() for encoding.\n
• train_test_split() to split data.\n
• SimpleImputer for missing value replacement.\n
• ColumnTransformer() for column preprocessing.\n
         """)

st.header("Results and Discussion:\n")
st.write("""
To test for our results, we prepared our target variable for testing and evaluated the performance of our trained neural network model using said data. As a result, we found our test loss to be approximately 0.121 and our test accuracy to be approximately 0.957. Because of our relatively low test loss as well as our relatively high test accuracy, we can conclude that our model performed well and is able to generalize from our training data to unseen data in our testing set. Additionally, we gathered both a confusion matrix and an accuracy and loss curve to better illustrate the process: \n
         
""")
st.image("confusion_matrix.png")
st.image("training_accuracy.png")
st.write("As seen from both charts, the addition of 8 epochs provided a strong influence on the effectiveness of the model. Other factors which lead to these accurate results include the addition of two hidden layers with ReLU activation functions inside of them, a sufficient amount of data for the neural network to work with (1728 different cars), and the preprocessing steps of imputation and one-hot encoding which were able to effectively handle missing values and categorical data accordingly. \n")
st.write("For the future, we wish to use similar strategies to create an Extreme Gradient Boosting model and a Support Vector Machine to see the differences in speed and accuracy.\n")
st.header("Midterm Contribution Table:")
st.dataframe(df, use_container_width=True)
st.header("Gantt Chart:")
st.write("https://docs.google.com/spreadsheets/d/1gD6TI02N_67U_YAIVeduaDORvfWMKicr/edit?gid=2146609855#gid=2146609855\n")

st.header("Works Cited")
st.write("""
[1] I. Bauer, L. Zavolokina, and G. Schwabe, “Is there a market for trusted car data?,” Electronic Markets, vol. 30, no. 2, pp. 211–225, Sep. 2019, doi: https://doi.org/10.1007/s12525-019-00368-5.\n
[2] Bohanec, M. (1988). Car Evaluation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48.\n
[3] A. Das Mou, P. K. Saha, S. A. Nisher, and A. Saha, “A Comprehensive Study of Machine Learning algorithms for Predicting Car Purchase Based on Customers Demands,” 2021 International Conference on Information and Communication Technology for Sustainable Development (ICICT4SD), Feb. 2021, doi: https://doi.org/10.1109/icict4sd50815.2021.9396868.\n
[4] S. Singh, S. Das, A. Sajwan, I. Singh, and A. Alok, “Car Recommendation System Using Customer Reviews,” *International Research Journal of Engineering and Technology (IRJET)*, vol. 9, no. 10, pp. 983–989, Oct. 2022
"""
)

