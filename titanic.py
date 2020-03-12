import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv("train.csv")

titanic_df.head()

titanic_df.info()

titanic_df["Fare"].mean()
titanic_df["Fare"].std()
titanic_df["Fare"].max()
titanic_df["Fare"].min()

sns.catplot("Pclass", data=titanic_df, kind="count", hue="Sex")

def male_female_child(passenger):
    age,sex = passenger

    if age < 16:
        return 'child'
    else:
        return sex

titanic_df['person']=titanic_df[['Age','Sex']].apply(male_female_child,axis=1)

titanic_df[0:10]
sns.catplot("Pclass", data=titanic_df, hue="person", kind="count")

titanic_df["Age"].hist(bins=70)

titanic_df["Age"].mean()
titanic_df['person'].value_counts()
sns.FacetGrid(titani_df, hue="Sex", aspect=4)
fig.map(sns.kdeplot, "Age", shade=True)

oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
