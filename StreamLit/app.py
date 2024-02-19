import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class App:
    def __init__(self):
        self.initialize()
    def run(self):
        if self.uploaded_file is not None:
            self.processing_on_dataset()
            self.apply_preprocessing()
            self.create_model()
        else:
            st.write('<h1><span style="color: red;">Please Choose your Dataset At First</span></h1>', unsafe_allow_html=True)
    def initialize(self):
        #st.write("""
        #             <script>
        #                 document.querySelector("[accept='']").setAttribute("accept", ".csv");
        #             </script>
        #         """, unsafe_allow_html=True)
        self.uploaded_file = st.sidebar.file_uploader("Upload the data file", type=["csv"],accept_multiple_files=False)        

        
        # Hocam amaç projeyi ne kadar geliştirdiğimiz de olduğu için ben de 3 tane algoritmayla sınırlı kalmayıp diğer algoritmaları
        # da ekledim ödeve
        self.classifier_name = st.sidebar.selectbox(
            'Select the Algorithm',
            ('KNN', 'SVM', "Naive Bayes", 'Random Forest',"XGBoost","Neural Network")
        )
        #self.normalization_name = st.sidebar.selectbox(
        #    'Select Normalization Method',
        #    ('Nothing',"Z-Score Normalization","Min-Max Scaling", "Robust Scaling")
        #)
    def processing_on_dataset(self):

        if self.uploaded_file is not None:
            st.title("Breast Cancer Dataset Analysis")
            data=pd.read_csv(self.uploaded_file)
            st.write('# Verisetinin ilk 10 satırı:', data.head(10))
            st.markdown("---")
            st.write('# Verisetinin sütunları:', data.columns)
            #st.table(data.columns)
            data.drop(['id','Unnamed: 32'], inplace= True, axis = 1)
            st.markdown("---")
            st.write('# Gereksiz sütunları sildikten sonra verisetinin son 10 satırı:', data.tail(10))
            st.markdown("---")
            #data.replace(0, 0.0, inplace=True)
            #data.dropna(inplace=True)
            data['diagnosis']=data['diagnosis'].map({'M': 1, 'B': 0})

            #text_html = '<h1> X<sub style="font-size: 20px;">train</sub> ve Y<sub style="font-size: 20px;">train</sub> </h1>'
            
            st.write("# Verilerin Kısa Analizi" )
            text_html_x = '## X değerleri:'
            self.X = data.drop('diagnosis', axis=1)
            st.write(text_html_x, self.X)
            
            #text_html_y = '<h1> Y değerleri: </h1>'
            self.y=data['diagnosis']
            st.write("### X'in Boyutu: ", self.X.shape)
            st.write("### Y'deki class sayısı: ", len(np.unique(self.y)))
            #st.write(text_html_y,self.y,unsafe_allow_html=True)
        
        
            st.markdown("---")
            st.write("# Correlation Matrix using Scatter Plot")

            column_names = self.X.columns.tolist()


            default_column_name = 'radius_mean'
            default_index = column_names.index(default_column_name)

            default_column_name1 = 'texture_mean'
            default_index1 = column_names.index(default_column_name1)
            
            
            x_column = st.selectbox('Select column for X-axis', options=self.X.columns,index=default_index)
            y_column = st.selectbox('Select column for Y-axis', options=self.X.columns,index=default_index1)


            plt.figure(figsize=(8, 6))
            scatterplot=sns.scatterplot(data=data, x=x_column, y=y_column, hue="diagnosis", palette='viridis', s=100,)
            plt.title('Correlation Matrix')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid(True)
            plt.legend(title="Diagnosis")
            legend_labels = scatterplot.get_legend().get_texts()
            
            
            
            
            for label in legend_labels:
                if label.get_text() == '1':
                    label.set_text('Malignant')
                if label.get_text() == '0':
                    label.set_text('Benign')
            st.pyplot(plt)
            
            st.markdown("---")
            
    def apply_preprocessing(self):
        #if self.normalization_name == 'Z-Score Normalization':
        #    scaler = StandardScaler()
        #    df_normalized = scaler.fit_transform(self.X)
        #    self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        #elif self.normalization_name == 'Min-Max Scaling':
        #    scaler=MinMaxScaler()
        #    df_normalized = scaler.fit_transform(self.X)
        #    self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        #elif self.normalization_name == 'Robust Scaling':
        #    scaler=RobustScaler()
        #    df_normalized = scaler.fit_transform(self.X)
        #    self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        #elif self.normalization_name == 'Nothing':
        #    pass
        
        if self.classifier_name == 'KNN':
            scaler=MinMaxScaler()
            df_normalized = scaler.fit_transform(self.X)
            self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        elif self.classifier_name == 'SVM':
            scaler = StandardScaler()
            df_normalized = scaler.fit_transform(self.X)
            self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        elif self.classifier_name == 'Naive Bayes':
            pass
            #scaler=RobustScaler()
            #df_normalized = scaler.fit_transform(self.X)
            #self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        elif self.classifier_name == 'Random Forest':
            pass
        elif self.classifier_name == "XGBoost":
            pass
        elif self.classifier_name == "Neural Network":
            scaler = StandardScaler()
            df_normalized = scaler.fit_transform(self.X)
            self.X = pd.DataFrame(df_normalized, columns=self.X.columns)

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            self.clf  = SVC()
        elif self.classifier_name == 'KNN':
            self.clf  = KNeighborsClassifier()
        elif self.classifier_name == 'Random Forest':
            self.clf  = RandomForestClassifier()
        elif self.classifier_name == "Naive Bayes":
            pass
        elif self.classifier_name == "XGBoost":
            self.clf = xgb.XGBClassifier()
        elif self.classifier_name == "Neural Network":
            self.clf = Sequential([
                Dense(15, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
                Dense(6, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
                Dense(1, activation='sigmoid')
            ])
            self.clf.compile(optimizer=Adam(learning_rate=0.001),
                          loss=BinaryCrossentropy(),
                          metrics=['accuracy'])
            
            
    def grid_search(self):
        if self.classifier_name == 'SVM':
            #param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
            self.grid_search1 = GridSearchCV(self.clf, param_grid, cv=5)
        elif self.classifier_name == 'KNN':
            param_grid = {'n_neighbors': [3,4, 5, 7]}
            self.grid_search1 = GridSearchCV(self.clf, param_grid, cv=5)
        elif self.classifier_name == 'Random Forest':
            #aralik=0.1
            #aralik_son=1.0
            param_grid = {
                #'n_estimators': [200, 300],
                'max_depth': range(4,6)
                #'min_samples_split': [1, 2]
                #'min_samples_leaf': [1, 2, 4]
                #'min_weight_fraction_leaf': [0.0] + [0.1 * i for i in range(0, int(aralik_son//aralik))]
            }
            self.grid_search1 = GridSearchCV(self.clf, param_grid, cv=5)
        elif self.classifier_name == "XGBoost":
            param_grid = {
                             'learning_rate': [0.1, 0.01, 0.001],
                             'max_depth': [4],
                             'n_estimators': [50, 100, 200]
                         }
            self.grid_search1 = GridSearchCV(estimator=self.clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
        elif self.classifier_name == "Naive Bayes":
            self.grid_search1=GaussianNB()
        
        
    def confusion_matrix(self):
        st.title('Confusion Matrix')

        cm = confusion_matrix(self.y_test, self.y_pred)


        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True,  fmt='d', cbar=True, cmap="YlGnBu")
        plt.xlabel("y_pred")
        plt.ylabel("y_test")
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        st.markdown("---")  
        
           
    def show_report(self):
        st.title('Reports for the algorithm')
        rounding=5
        acc =     round(accuracy_score(self.y_test, self.y_pred),rounding)
        f1=       round(f1_score(self.y_test, self.y_pred),rounding)
        precision=round(precision_score(self.y_test, self.y_pred),rounding)
        recall=   round(recall_score(self.y_test, self.y_pred),rounding)
        

        #st.write(f'Classifier = {self.classifier_name}')
        #st.write(f'Accuracy Score: {acc:.3f}')
        #st.write(f'F1 Score: {f1:.3f}')
        #st.write(f'Precision Score: {precision:.3f}')
        #st.write(f'Recall Score: {recall:.3f}')
        st.write(f'##### Classifier = {self.classifier_name}')
        st.write(f'##### Accuracy Score: ',acc)
        st.write(f'##### F1 Score: ',f1)
        st.write(f'##### Precision Score: ',precision)
        st.write(f'##### Recall Score: ',recall)
        
        if self.classifier_name=="Neural Network":
            pass

        else:
            st.write('## Best Parameters')
            if self.classifier_name == "Naive Bayes":
                st.write("##### GridSearch wasn't applied to Naive Bayes Algorithm. Because there is no hyperparameter.")
            else:
                for key, value in self.grid_search1.best_params_.items():
                    st.write(f"##### {key}: {value}")
        
        
    def create_model(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1234)
        #st.write('xtest:', len(X_test))
        self.get_classifier()
        if self.classifier_name=="Neural Network":
            self.clf.fit(self.X_train, self.y_train, epochs=200, verbose=1, validation_split=0.1)
            y_pred = self.clf.predict(self.X_test)
            #y_pred = tf.math.sigmoid(y_pred)
            self.y_pred = (y_pred > 0.5)

        else:
            self.grid_search()
            self.grid_search1.fit(self.X_train, self.y_train)
            self.y_pred = self.grid_search1.predict(self.X_test)

        
        self.confusion_matrix()
        self.show_report()
 
