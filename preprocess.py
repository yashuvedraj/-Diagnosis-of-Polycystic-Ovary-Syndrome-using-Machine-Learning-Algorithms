import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def process():
    
    file_path_with_infertility="PCOS_infertility.csv"
    file_path_without_infertility="PCOS_data_without_infertility.xlsx"

    PCOS_inf = pd.read_csv(file_path_with_infertility)
    PCOS_woinf = pd.read_excel(file_path_without_infertility, sheet_name="Full_new",engine='openpyxl',)
    data = pd.merge(PCOS_woinf,PCOS_inf, on='Patient File No.', suffixes={'','_y'},how='left')

    #Dropping the repeated features after merging
    data =data.drop(['Unnamed: 44', 'Sl. No_y', 'PCOS (Y/N)_y', '  I   beta-HCG(mIU/mL)_y',
           'II    beta-HCG(mIU/mL)_y', 'AMH(ng/mL)_y'], axis=1)

    #Taking a look at the dataset
    data.head()
    data.info()
    data["AMH(ng/mL)"].head()
    data["II    beta-HCG(mIU/mL)"].head()
    #Dealing with categorical values.
    #In this database the type objects are numeric values saved as strings.
    #So I am just converting it into a numeric value.

    data["AMH(ng/mL)"] = pd.to_numeric(data["AMH(ng/mL)"], errors='coerce')
    data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors='coerce')

    #Dealing with missing values. 
    #Filling NA values with the median of that feature.

    data['Marraige Status (Yrs)'].fillna(data['Marraige Status (Yrs)'].median(),inplace=True)
    data['II    beta-HCG(mIU/mL)'].fillna(data['II    beta-HCG(mIU/mL)'].median(),inplace=True)
    data['AMH(ng/mL)'].fillna(data['AMH(ng/mL)'].median(),inplace=True)
    data['Fast food (Y/N)'].fillna(data['Fast food (Y/N)'].median(),inplace=True)

    #Clearing up the extra space in the column names (optional)

    data.columns = [col.strip() for col in data.columns]
    data.describe()
    corrmat = data.corr()
    #Having a look at features bearing significant correlation
    plt.subplots(figsize=(18,18))
    sns.heatmap(corrmat,cmap="Pastel1", square=True);
    corrmat["PCOS (Y/N)"].sort_values(ascending=False)
    plt.figure(figsize=(12,12))
    k = 12 #number of variables with positive for heatmap
    l = 3 #number of variables with negative for heatmap
    cols_p = corrmat.nlargest(k, "PCOS (Y/N)")["PCOS (Y/N)"].index 
    cols_n = corrmat.nsmallest(l, "PCOS (Y/N)")["PCOS (Y/N)"].index
    cols = cols_p.append(cols_n) 

    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True,cmap="Pastel1", annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    # Length of menstrual phase in PCOS vs normal 
    color = ["teal", "plum"]
    fig=sns.lmplot(data=data,x="Age (yrs)",y="Cycle length(days)", hue="PCOS (Y/N)",palette=color)
    plt.show()

    # Pattern of weight gain (BMI) over years in PCOS and Normal. 
    fig= sns.lmplot(data =data,x="Age (yrs)",y="BMI", hue="PCOS (Y/N)", palette= color )
    plt.show()
    # cycle IR wrt age 
    sns.lmplot(data =data,x="Age (yrs)",y="Cycle(R/I)", hue="PCOS (Y/N)",palette=color)
    plt.show()
    # Distribution of follicles in both ovaries. 
    sns.lmplot(data =data,x='Follicle No. (R)',y='Follicle No. (L)', hue="PCOS (Y/N)",palette=color)
    plt.show()
    features = ["Follicle No. (L)","Follicle No. (R)"]
    for i in features:
        sns.swarmplot(x=data["PCOS (Y/N)"], y=data[i], color="black", alpha=0.5 )
        sns.boxenplot(x=data["PCOS (Y/N)"], y=data[i], palette=color)
        plt.show()
        features = ["Age (yrs)","Weight (Kg)", "BMI", "Hb(g/dl)", "Cycle length(days)","Endometrium (mm)" ]
    for i in features:
        sns.swarmplot(x=data["PCOS (Y/N)"], y=data[i], color="black", alpha=0.5 )
        sns.boxenplot(x=data["PCOS (Y/N)"], y=data[i], palette=color)
        plt.show()
    #Assiging the features (X)and target(y)
    data.to_csv("Preprecesed_dataset.csv")

