### SpaceShip Titanic Supervised Machine Learning Project: CSCA 5622 ###
For my Final project I am going to execute a supervised learning analysis on a kaggle data set https://www.kaggle.com/competitions/spaceship-titanic/data <br>
Here I am using the "Spaceship Titanic" data from an ongoing Kaggle competition where we will predict which passengers are transported to an alternate dimension.
This project includes 3 deliverables: 
* This Juypter notebook - including: problem Decription, EDA procedure, analysis(model building and training), result, and discussion/conclusion
* A video presentation available on Youtube - Explaining what problem we solved, my ML approach and method, and Final Result
    * link here
* and a public github repository - https://github.com/GHeart01/SpaceShip-Titanic-Supervised-ML-Project

#### Table of Contents

- [Problem Description](#Problem-Description)
- [EDA](#Exploratory-Data-Analysis-(EDA)-Procedure)
- [Analysis](#Analysis)
- [Result](#Result)
- [Discussion](#Discussion)
- [Citation](#Citation)
#### Problem Description
First lets take a brief overview of our Machine Learning (ML) problem. In this scenario we explore a play on a well known disaster, The Titanic, in which a "ground truth" for each passenger is determined. Our goal is less gruesome, as instead of life and death on the spaceship tianic we will determine whether almost 13,000 passengers on board will be transported to three habitable exoplanets orbiting nearby stars. Unfortunately while rounding our first destination, the Spaceship Titanic collided with a spacetime anomaly. While the ship stayed intact, almost half of the passengers were transported to an alternate dimension!  



<p align="center">
<img src="assets/alpha-c.jpg" width="700" height="350">
<br>
Spacetime Anomaly
<p>


To resue the lost passengers we must use Machine Learning to predict which passengers were transported by the anomaly using records recovered from the spaceship's damaged computer system!
### Exploratory Data Analysis (EDA) Procedure
My task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help  make these predictions, I use a given a set of personal records recovered from the ship's damaged computer system.

|Variable |	Description|
|:---------|:-------------|
|PassengerId 	|A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.|
|HomePlanet |	The planet the passenger departed from, typically their planet of permanent residence.|
|CryoSleep |	 Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.|
|Cabin |	The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.|
|Destination |	The planet the passenger will be debarking to.|
|Age| 	 The age of the passenger.|
|VIP |	Whether the passenger has paid for special VIP service during the voyage.|
|RoomService, FoodCourt, ShoppingMall, Spa, VRDeck |	Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.|
|Name |	The first and last names of the passenger.|
|Transported |	Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.|

# Here I will be using tensorflow, we did not use TF in our class lecture, but it contains libraries that make similar to sklearn
import tensorflow as tf
import tensorflow_decision_forests as tfdf

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
### Analysis
### Result
### Discussion
### Citation
Addison Howard, Ashley Chow, and Ryan Holbrook. Spaceship Titanic. https://kaggle.com/competitions/spaceship-titanic, 2022. Kaggle.

### Clone this repo
git clone https://github.com/GHeart01/SpaceShip-Titanic-Supervised-ML-Project.git