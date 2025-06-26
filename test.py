# Here I will be using tensorflow, we did not use TF in our class lecture, but it contains libraries that make similar to sklearn
import tensorflow as tf
import tensorflow_decision_forests as tfdf

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

# Load a dataset into a Pandas Dataframe
dataset_df = pd.read_csv('spaceship-titanic/train.csv')
print("Full train dataset shape is {}".format(dataset_df.shape))

print("\nHead")
print(dataset_df.head(5))
print("\nDescribe")
print(dataset_df.describe())
print("\nInfo")
print(dataset_df.info())


# transported count bar plot
plot_df = dataset_df.Transported.value_counts()
plot_df.plot(kind="bar")
plt.title("Transported Count (Target Variable)")
plt.show()  


# histograms
fig, ax = plt.subplots(5,1,  figsize=(10, 10))
plt.subplots_adjust(top = 2)

sns.histplot(dataset_df['Age'], color='b', bins=50, ax=ax[0]);
sns.histplot(dataset_df['FoodCourt'], color='b', bins=50, ax=ax[1]);


sns.histplot(dataset_df['ShoppingMall'], color='b', bins=50, ax=ax[2]);
sns.histplot(dataset_df['Spa'], color='b', bins=50, ax=ax[3]);

sns.histplot(dataset_df['VRDeck'], color='b', bins=50, ax=ax[4]);
plt.show()  
