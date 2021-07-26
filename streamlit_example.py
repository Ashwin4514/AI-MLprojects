import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Streamlit Example")

st.write("""
# Explore Different Classifiers

Which one is the best?
""")

dataset_names= st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))

classifier_name= st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))

def get_Dataset(dataset_name):
    if(dataset_name=="Iris"):
       data= datasets.load_iris()
    elif(dataset_name=="Breast Cancer"):
        data= datasets.load_breast_cancer()

    else:
        data= datasets.load_wine()
    X= data.data
    y= data.target

    return X,y


X,y = get_Dataset(dataset_names)
st.write("Shape of Dataset",X.shape)
st.write("Number of classes", len(np.unique(y)))

def parameters(classifier_name):
    params= dict()
    if(classifier_name=="KNN"):
        K= st.sidebar.slider("K", 1, 15)
        params["K"]= K
    elif(classifier_name=="SVM"):
        C= st.sidebar.slider("C", 0.01, 10.0)
        params["C"]= C
    else:
        max_depth= st.sidebar.slider("Max_Depth", 2, 15)
        n_estimators= st.sidebar.slider("n_estimators", 1,100)
        params["Max_Depth"]= max_depth
        params["n_estimators"]= n_estimators
    return params

params= parameters(classifier_name)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf= KNeighborsClassifier(n_neighbors= params["K"])
       
    elif clf_name == "SVM":
        clf= SVC(C= params["C"])
  
    else:
        clf= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=12)

    return clf

classifier= get_classifier(classifier_name,params)

#Classification

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.1,random_state=12)

classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)

acc= accuracy_score(y_test,y_pred)
st.write(f"Classifier= {classifier_name}")
st.write(f"accuracy= {acc}")

#PLOT
pca=PCA(2)
X_projected= pca.fit_transform(X)


x1= X_projected[:,0]
x2= X_projected[:,1]

fig= plt.figure()
plt.scatter(x1,x2,c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)
