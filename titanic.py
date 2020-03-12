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

fig = sns.FacetGrid(titanic_df, hue="Sex", aspect=4)
fig.map(sns.kdeplot, "Age", shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df, hue="person", aspect=4)
fig.map(sns.kdeplot, "Age", shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df, hue="Pclass", aspect=4)
fig.map(sns.kdeplot, "Age", shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

deck = titanic_df["Cabin"].dropna()
deck.head()
levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ["Cabin"]
cabin_df = cabin_df.sort_values("Cabin")
sns.catplot("Cabin", data = cabin_df, palette = "winter_d", kind = "count")

cabin_df = cabin_df[cabin_df.Cabin != "T"]
sns.catplot("Cabin", data = cabin_df, palette = "summer", kind = "count")

sns.catplot("Embarked", data = titanic_df, hue = "Pclass", kind = "count", order = ['C', 'Q', 'S'])


titanic_df["Alone"] = titanic_df.SibSp + titanic_df.Parch
titanic_df["Alone"].loc[titanic_df["Alone"] > 0] = "With Family"
titanic_df["Alone"].loc[titanic_df["Alone"] == 0] = "Alone"

titanic_df.head()

sns.catplot("Alone", data = titanic_df, palette = "Blues", kind = "count", order = ["Alone", "With Family"])

titanic_df["Survivor"] = titanic_df.Survived.map({0:"no", 1:"yes"})
titanic_df.head()
sns.catplot("Survivor", data = titanic_df, palette = "Set1", kind = "count")

sns.catplot("Pclass", "Survived", data = titanic_df, hue = "person", kind = "point")

sns.lmplot("Age", "Survived", hue = "Pclass", data = titanic_df, palette = "winter")
generations = [10, 20, 30, 40, 50, 60, 70, 80]
sns.lmplot("Age", "Survived", data = titanic_df, hue = "Pclass", palette = "winter", x_bins = generations)
sns.lmplot("Age", "Survived", hue = "Sex", data = titanic_df, palette = "winter", x_bins = generations)
