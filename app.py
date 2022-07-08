import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')
#---------------------------------#
# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)

    # Build lazy model
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    # st.markdown('**1.2. Check**')
    # st.write('X_train')
    # st.write(X_train)
    # st.write('X_test')
    # st.write(X_test)
    # st.write('Y_train')
    # st.write(y_train)
    # st.write('Y_test')
    # st.write(y_test)


    if type == 'Regression':
        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models_train,predictions_train = reg.fit(X_train, X_train, y_train, y_train)
        models_test,predictions_test = reg.fit(X_train, X_test, y_train, y_test)
    else:
        clf = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
        models_train,predictions_train = clf.fit(X_train, X_train, y_train, y_train)
        models_test,predictions_test = clf.fit(X_train, X_test, y_train, y_test)


    st.subheader('2. Table of Model Performance')

    # st.write('Training set')
    st.write(predictions_train)
    # st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

    # st.write('Test set')
    # st.write(predictions_test)
    # st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    st.subheader('3. Plot of Model Performance')
    if type == 'Regression':
        with st.markdown('**R-squared**'):
            # Tall
            predictions_train["R-Squared"] = [0 if i < 0 else i for i in predictions_train["R-Squared"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=predictions_train.index, x="R-Squared", data=predictions_train)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_train.index, y="R-Squared", data=predictions_train)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**RMSE**'):
            # Tall
            predictions_train["RMSE"] = [i if i > 50 else i for i in predictions_train["RMSE"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=predictions_train.index, x="RMSE", data=predictions_train)
        st.markdown(imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_train.index, y="RMSE", data=predictions_train)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-rmse-wide.pdf'), unsafe_allow_html=True)
    else:
        with st.markdown('**Accuracy**'):
            # Tall
            #predictions_train["Accuracy"] = [i if i < 0 else i for i in predictions_train["Accuracy"] ]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=predictions_train.index, x="Accuracy", data=predictions_train)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt,'plot-accuracy-tall.pdf'), unsafe_allow_html=True)
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_train.index, y="Accuracy", data=predictions_train)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-accuracy-wide.pdf'), unsafe_allow_html=True)

    with st.markdown('**Calculation time**'):
        # Tall
        predictions_train["Time Taken"] = [0 if i < 0 else i for i in predictions_train["Time Taken"] ]
        plt.figure(figsize=(3, 9))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(y=predictions_train.index, x="Time Taken", data=predictions_train)
    st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
        # Wide
    plt.figure(figsize=(9, 3))
    sns.set_theme(style="whitegrid")
    ax3 = sns.barplot(x=predictions_train.index, y="Time Taken", data=predictions_train)
    plt.xticks(rotation=90)
    st.pyplot(plt)
    st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)

# Download CSV data
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
st.write("""
# Group 2 - Machine Learning Project
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

# Sidebar - Specify Type of Problem
with st.sidebar.header('3. Select Type'):
    type = st.sidebar.radio("Type of Problem",('Regression', 'Classification'))

# Sidebar - Specify Target Variable
with st.sidebar.header('4. Specify Target Variable'):
    target = st.sidebar.text_input('Target Variable Name','')


#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    if target is not None:
        if target in list(df.columns):
            Y = df.pop(target)
            X = df
            df = pd.concat( [X,Y], axis=1 )
            build_model(df)
        else:
            st.info('Enter Correct Target Variable.')
    else:
        st.info('Enter Target Variable.')
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        if type == 'Classification':
            # Breast Cancer dataset
            breast_cancer = load_breast_cancer()
            #X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
            #Y = pd.Series(breast_cancer.target, name='response')
            X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            Y = pd.Series(breast_cancer.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            df = pd.concat( [X,Y], axis=1 )

            st.markdown('The Breast Cancer dataset is used as the example.')
            st.write(df.head(5))
        else:
            # Boston housing dataset
            boston = load_boston()
            #X = pd.DataFrame(boston.data, columns=boston.feature_names)
            #Y = pd.Series(boston.target, name='response')
            X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            Y = pd.Series(boston.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            df = pd.concat( [X,Y], axis=1 )

            st.markdown('The Boston housing dataset is used as the example.')
            st.write(df.head(5))

        build_model(df)
