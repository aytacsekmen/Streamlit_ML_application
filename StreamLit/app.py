import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class App:
    def __init__(self):
        self.dataset_name = None
        self.classifier_name = None
        self.Init_Streamlit_Page()

        self.params = dict()
        self.clf = None
        self.X, self.y = None, None
    def run(self):
        if self.uploaded_file is not None:
            self.get_dataset()
            self.apply_preprocessing()
            #self.add_parameter_ui()
            self.generate()
        else:
            st.write('<h1><span style="color: red;">Please Choose your Dataset First</span></h1>', unsafe_allow_html=True)
    def Init_Streamlit_Page(self):
        #st.title('Streamlit Example')
#
        #st.write("""
        ## Explore different classifier and datasets
        #Which one is the best?
        #""")
        st.write("""
                     <script>
                         document.querySelector("[accept='']").setAttribute("accept", ".csv");
                     </script>
                 """, unsafe_allow_html=True)
        self.uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv"],accept_multiple_files=False)
        #self.dataset_name = st.sidebar.selectbox(
        #    'Select Dataset',
        #    ('Breast Cancer',)
        #)
        

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', "Naive Bayes", 'Random Forest')
        )
        self.normalization_name = st.sidebar.selectbox(
            'Select Normalization Method',
            ('Nothing',"Z-Score Normalization","Min-Max Scaling", "Robust Scaling")
        )
    def get_dataset(self):

        if self.uploaded_file is not None:
            st.title("Breast Cancer Dataset Analysis")
            data=pd.read_csv(self.uploaded_file)
            st.write('# Verisetinin ilk 10 satırı:', data.head(10))
            st.markdown("---")
            st.write('# Verisetinin sütunları:', data.columns)
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
        if self.normalization_name == 'Z-Score Normalization':
            scaler = StandardScaler()
            df_normalized = scaler.fit_transform(self.X)
            self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        elif self.normalization_name == 'Min-Max Scaling':
            scaler=MinMaxScaler()
            df_normalized = scaler.fit_transform(self.X)
            self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        elif self.normalization_name == 'Robust Scaling':
            scaler=RobustScaler()
            df_normalized = scaler.fit_transform(self.X)
            self.X = pd.DataFrame(df_normalized, columns=self.X.columns)
        elif self.normalization_name == 'Nothing':
            pass
            
    def add_parameter_ui(self):
        if self.classifier_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 15.0)
            self.params['C'] = C
            
        elif self.classifier_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            
            self.params['K'] = K
        elif self.classifier_name == 'Random Forest':
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            self.params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            self.params['n_estimators'] = n_estimators
        #elif self.classifier_name == "Naive Bayes":
            #max_depth = st.sidebar.slider('max_depth', 2, 15)
            #self.params['max_depth'] = max_depth
            #n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            #self.params['n_estimators'] = n_estimators

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            #self.clf  = SVC(C=self.params['C'])
            self.clf  = SVC()
        elif self.classifier_name == 'KNN':
            #self.clf  = KNeighborsClassifier(n_neighbors=self.params['K'])
            self.clf  = KNeighborsClassifier()
        elif self.classifier_name == 'Random Forest':
            #self.clf  = RandomForestClassifier(n_estimators=self.params['n_estimators'],
                #max_depth=self.params['max_depth'], random_state=1234)
            self.clf  = RandomForestClassifier()
        elif self.classifier_name == "Naive Bayes":
            self.clf  = GaussianNB()
            
            
            
            
    def grid_search(self):
        if self.classifier_name == 'SVM':
            #param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
            self.grid_search1 = GridSearchCV(self.clf, param_grid, cv=5)
        elif self.classifier_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7]}
            self.grid_search1 = GridSearchCV(self.clf, param_grid, cv=5)
        elif self.classifier_name == 'Random Forest':
            aralik=0.1
            aralik_son=1.0
            param_grid = {
                #'n_estimators': [100, 200, 300],
                'max_depth': range(2,4)
                #'min_samples_split': [2, 5, 10],
                #'min_samples_leaf': [1, 2, 4]
                #'min_weight_fraction_leaf': [0.0] + [0.1 * i for i in range(0, int(aralik_son//aralik))]
            }
            self.grid_search1 = GridSearchCV(self.clf, param_grid, cv=5)
        elif self.classifier_name == "Naive Bayes":
            self.grid_search1=GaussianNB()
        
        
        
        
    def generate(self):
        self.get_classifier()
        #### CLASSIFICATION ####
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1234)
        #st.write('xtest:', len(X_test))
        self.grid_search()
        self.grid_search1.fit(X_train, y_train)
        y_pred = self.grid_search1.predict(X_test)
        
        

        st.title('Confusion Matrix')
        
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True,  fmt='d', cbar=False, cmap="YlGnBu")
        plt.xlabel("y_pred")
        plt.ylabel("y_test")
        plt.title('Confusion Matrix')
        st.pyplot(plt)
        st.markdown("---")

        st.title('Reports for the algorithm')
        rounding=5
        acc =     round(accuracy_score(y_test, y_pred),rounding)
        f1=       round(f1_score(y_test, y_pred),rounding)
        precision=round(precision_score(y_test, y_pred),rounding)
        recall=   round(recall_score(y_test, y_pred),rounding)

        #st.write(f'Classifier = {self.classifier_name}')
        #st.write(f'Accuracy Score: {acc:.3f}')
        #st.write(f'F1 Score: {f1:.3f}')
        #st.write(f'Precision Score: {precision:.3f}')
        #st.write(f'Recall Score: {recall:.3f}')
        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy Score: ',acc)
        st.write(f'F1 Score: ',f1)
        st.write(f'Precision Score: ',precision)
        st.write(f'Recall Score: ',recall)
        
        
        #### PLOT DATASET ####
        # Project the data onto the 2 primary principal components
        #pca = PCA(2)
        #X_projected = pca.fit_transform(self.X)
        #x1 = X_projected[:, 0]
        #x2 = X_projected[:, 1]
        #fig = plt.figure()
        #plt.scatter(x1, x2,
        #        c=self.y, alpha=0.8,
        #        cmap='viridis')
        #st.pyplot(fig)