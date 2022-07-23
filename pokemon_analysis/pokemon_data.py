#pokemon data analysis with weedle cave data from kaggle
#analysis is to fing the best pokemon for battle

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

import random
random.seed(1)

#dataset
pokemon=pd.read_csv('pokemon.csv')
#print(pokemon.head())

# we need to change # column name-will otherwise comment code

pokemon = pokemon.rename(index=str, columns={"#": "Number"})
combat = pd.read_csv("combats.csv")
#print(pokemon.head())

print("pokemon:",pokemon.shape)
print("combat:",combat.shape)

#missing values
print(pokemon.isnull().sum())
print(combat.isnull().sum())

#one name missing
#type 2 is missing for 386 pokemon
#missing,62 is primeape-->google
pokemon['Name'][62]="Primeape"

#---------------------------------------

#calculate % wins and add them to the dataset
total_Wins=combat.Winner.value_counts() #add calculation of %
numberOfWins=combat.groupby('Winner').count()  #no of wins for each pokemon

countByFirst=combat.groupby('Second_pokemon').count()
countBySecond=combat.groupby('First_pokemon').count()

print("Looking at the dimensions of our dataframes")
print("Count by first winner shape: " + str(countByFirst.shape))
print("Count by second winner shape: " + str(countBySecond.shape))
print("Total Wins shape : " + str(total_Wins.shape))
print(countByFirst.head())

'''We can see that the number of dimensions is different 
in the total wins. This can only mean there is one pokemon 
that was unable to win during it's fights. Lets find the 
pokemon that did not win a single fight.'''
find_loosing=np.setdiff1d(countByFirst.index.values,numberOfWins.index.values)-1 #offset as number/index are off by 1
losing_pokemon=pokemon.iloc[find_loosing[0],]
print(losing_pokemon)

'''
Poor Shuckle
just add shuckle to the data as it will make merging easier
'''
'''
since there is a pokemon with no wins
there may be one with no battles
lets check
'''
#feature enginerring

numberOfWins=numberOfWins.sort_index()
numberOfWins['Total Fights']=countByFirst.Winner+countBySecond.Winner
numberOfWins['Win Percentage']=numberOfWins.First_pokemon/numberOfWins['Total Fights']

# merge the winning dataset and the original pokemon dataset
results2 = pd.merge(pokemon, numberOfWins, right_index = True, left_on='Number')
results3 = pd.merge(pokemon, numberOfWins, left_on='Number', right_index = True, how='left')

# We can look at the difference between the two datasets to see which pokemon never recorded a fight
#missing_Pokemon = np.setdiff1d(pokemon.index.values, results3.index.values)
#subset the dataframe where pokemon win percent is NaN
print(results3[results3['Win Percentage'].isnull()])

#pokemon in worst 10
print(results3[np.isfinite(results3['Win Percentage'])].sort_values(by=['Win Percentage']).head(10))
#pokemon in best 10
print(results3[np.isfinite(results3['Win Percentage'])].sort_values(by=['Win Percentage']).tail(10))

results3.describe()
#----------------------------------------------------------------
#data visualization

import matplotlib.pyplot as plt

#total pokemon by type 1
sns.set_color_codes("pastel")
ax=sns.countplot(x="Type 1",hue="Legendary",data=results3)
plt.xticks(rotation=90)
plt.xlabel('Type 1')
plt.ylabel('Total ')
plt.title("total pokemon by type 1")
plt.show()

#total pokemon by type 2
ax = sns.countplot(x="Type 2", hue="Legendary", data=results3)
plt.xticks(rotation= 90)
plt.xlabel('Type 2')
plt.ylabel('Total ')
plt.title("Total Pokemon by Type 2")
plt.show()

#Lets aggregate our win percentage by type 1.
results3.groupby('Type 1').agg({"Win Percentage": "mean"}).sort_values(by = "Win Percentage")

'''
We can further break down the data by looking at type 
by generation, legendary by generation, stats by type, 
stats by generation, stats by lengendary and so on.
'''

'''
I broke up the data into a smaller subset for this section. 
I defined the indepenent variables to be [ 'HP', 'Attack', 
'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'] and the dependent 
variable to be ['Win Percentage']. In this section, 
I will be exploreing these relationships as well as some 
other visualizations that will better explain the trends in the
 Pokeverse. I keep 'Type 1' in the data because later on I 
 want to see how the relationships break down by type
'''

col = ['Type 1','HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Win Percentage']

#pairplot and pairplot
sns.pairplot(results3.loc[:,col].dropna())

g = sns.PairGrid(results3.loc[:,col], diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(sns.regplot)
g.map_diag(sns.kdeplot, lw=3)

#correlation table
print(results3.loc[:,col].corr())
# https://datascience.stackexchange.com/questions/10459/calculation-and-visualization-of-correlation-matrix-with-pandas
#This function was taken from the link above 
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 50)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Pokemon Feature Correlation')
    labels=['Type 1','HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Win %']
    ax1.set_xticklabels(labels,fontsize=7)
    ax1.set_yticklabels(labels,fontsize=7)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.00,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.8,.85,.90,.95,1])
    plt.show()

correlation_matrix(results3.loc[:,col])

#f, (ax1, ax2) = plt.subplots(1,2)
sns.regplot(x="Speed", y="Win Percentage", data=results3, logistic=True).set_title("Speed vs Win Percentage")
sns.lmplot(x="Speed", y="Win Percentage", data=results3, hue = 'Type 1',  logistic=True)#.set_title("Speed vs Win Percentage")

ax = sns.regplot(x="Attack", y="Win Percentage", data=results3).set_title("Attack vs Win Percentage")
sns.lmplot(x="Attack", y="Win Percentage", data=results3, hue = 'Type 1',fit_reg =False)

'''
Exploratory Data Analysis Conclusions
This is where we communicate all the insight we have developed into a concise manner. Remeber, its not enought to simply state the results. Try and relate it back to the Team Rocket business model.

Water, normal, bug and grass are the most common type 1 and flying, ground and poison are the most common type 2.
Team Rocket should have pokemon in their battle squads to counter these types. Pokemon Type Weaknesses
The Pokemon type that win the most are flying, dragon, electric and dark. The Pokemon type that win the least are fairy, rock, steel, poison.
While this seems stright forward given that we have looked at the data, it might not always be apparent to those who have not. Communicating to Team Rocket, "Hey, these are the kinds of Pokemon (flying, dragon, electric and dark) you should be spending your resources on because they win. If you have these pokemon (fairy, rock, steel, poison) you should avoid wasting resources and release any you have into the wild so you can reduce your overhead cost.
Speed and Attack!!
Defense doesn't win championships in the Pokeverse. We need powerful attacks and quickness. If we look back at the top 10 most winning pokemon, all have speeds over 100+ and attacks over 100+ (except for Greninja's attack)
As data consultants, it might not be enough to only work with this dataset. It our job to ask "what else is missing?" or "what further analysis can we do given more data?"

What else are we missing and further analysis
I'm not sure if this data is similar to the video game but maybe pokemon level (ie 1-100) plays a role in this. A level 1 Pikachu would lose to a level 50 Blastoise, regardless of the fact Pikachu has the type advantage.
Can we get the data of pokemon battle squads from previous tournaments. We can start to analyze what kinds of pokemon are used at the competative level.
Geographic location of where to find the different Pokemon would be a tremendous help into advising Team Rocket.
We know from the TV show that Pokemon have their own personalities, some pokemon are stubborn, don't work well with others or listen to their masters. Getting textual review data about what people think about each pokemon could be helpful in understanding how Team Rocket could train them.
'''

#modeling
#Multiple Linear Regression, Polynomial Regression, SVM, Decision Tree Regression, Random Forest, XGBoost

dataset=results2

#remove rows with NA values because it will cause errors when fitting to the model
dataset.dropna(axis=0, how='any')
# Splitting the dataset into the Training set and Test set
X = dataset.iloc[:, 5:11].values
y = dataset.iloc[:, 15].values

# Encoding categorical data (if there is some)
# In this case it could be pokemon type
#'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder = LabelEncoder()
#X[:, 3] = labelencoder.fit_transform(X[:, 3])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()'''

# Avoiding the Dummy Variable Trap
#X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#multiple linear regression
def ml_linearreg(X_train, X_test, y_train, y_test):
    # Fitting Multiple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print(regressor.score(X_train, y_train))
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    # Validating the results
    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    mae = mean_absolute_error(y_test, y_pred)
    #print("Mean Absolute Error: " + str(mae))
    return mae

ml_linearreg(X_train, X_test, y_train, y_test)

#SVM
# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)
def ml_svm(X_train, X_test, y_train, y_test):
    # Fitting SVR to the dataset
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'linear')
    regressor.fit(X_train, y_train)
    print(regressor.score(X_train, y_train))

    #Predict Output
    y_pred= regressor.predict(X_test)
    #y_pred = sc_y.inverse_transform(y_pred)

    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    mae = mean_absolute_error(y_test, y_pred)
    #print("Mean Absolute Error: " + str(mae))
    return mae

ml_svm(X_train, X_test, y_train, y_test)

#decesion tree
#feature scaling not needed

def ml_decisiontree(X_train, X_test, y_train, y_test):
    # Fitting Decision Tree Regression to the dataset
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    print(regressor.score(X_train, y_train))

    # Predicting a new result
    y_pred = regressor.predict(X_test)

    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    mae = mean_absolute_error(y_test, y_pred)
    #print("Mean Absolute Error: " + str(mae))
    return mae

ml_decisiontree(X_train, X_test, y_train, y_test)

#random forest
# no feature scaling needed
def ml_randomforest(X_train, X_test, y_train, y_test):
    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
    print(regressor.score(X_train, y_train))

    # Predicting a new result
    y_pred = regressor.predict(X_test)

    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    mae = mean_absolute_error(y_test, y_pred)
    #print("Mean Absolute Error: " + str(mae))
    return mae

ml_randomforest(X_train, X_test, y_train, y_test)
    
# Visualising the Random Forest Regression results (higher resolution)
#X_grid = np.arange(min(X), max(X), 0.01)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (Random Forest Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()

#xgboost
def ml_xgboost(X_train, X_test, y_train, y_test):
    import xgboost
    # fitting XGBoost to training set
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
    xgb.fit(X_train,y_train)
    print(xgb.score(X_train, y_train))
    # Prediction
    y_pred = xgb.predict(X_test)
    #print(explained_variance_score(y_pred ,y_test))
    from sklearn.metrics import mean_absolute_error
    from math import sqrt
    mae = mean_absolute_error(y_test, y_pred)
    #print("Mean Absolute Error: " + str(mae))
    return mae

ml_xgboost(X_train, X_test, y_train, y_test)

#store all the ML results in an array
all_stats = [ml_linearreg(X_train, X_test, y_train, y_test), ml_svm(X_train, X_test, y_train, y_test), ml_decisiontree(X_train, X_test, y_train, y_test), ml_randomforest(X_train, X_test, y_train, y_test), ml_xgboost(X_train, X_test, y_train, y_test)]
print(all_stats)

'''
Principle component analysis (PCA) is a dimensionality reduction technique. It uses linear algebra to tranform the data into a new space of principle components. Each principle component explains some variance of the dataset. The goal of this technique is to reduce the amount of features we are using for our model and simplify. The principle components consist of component loadings. The loadings are the correlation coefficients between the variables and factors.
'''
#PCA

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
#pca = PCA(n_components = None)
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# Provides a vector of the variance explained by each component
explained_variance = pca.explained_variance_ratio_
print("This is the variance explained by the principle components")
print(explained_variance)

#loadings vectors
#print(pca.components_.T * np.sqrt(pca.explained_variance_))

'''
One you look at how much variance each independent variable provide, decide how many components you want for the model. The more components the more variance the model will have. Re-run the code but change n_components from 2 to the desired number.

Run the machine learning algorithms using the data after PCA.
'''
# run PCA transformed data on ML algos
PCA = [ml_linearreg(X_train, X_test, y_train, y_test), ml_svm(X_train, X_test, y_train, y_test), ml_decisiontree(X_train, X_test, y_train, y_test), ml_randomforest(X_train, X_test, y_train, y_test), ml_xgboost(X_train, X_test, y_train, y_test)]
#PCA
#ml_linearreg(X_train, X_test, y_train, y_test)
#ml_svm(X_train, X_test, y_train, y_test)
#ml_decisiontree(X_train, X_test, y_train, y_test)
#ml_randomforest(X_train, X_test, y_train, y_test)
#ml_xgboost(X_train, X_test, y_train, y_test)

# reduce the features to only speed and attack. 
dataset = results2
dataset.dropna(axis=0, how='any')
# Splitting the dataset into the Training set and Test set
X = dataset.loc[:, ['Attack','Speed']].values
y = dataset.loc[:, ['Win Percentage']].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#ml_linearreg(X_train, X_test, y_train, y_test)
#ml_svm(X_train, X_test, y_train, y_test)
#ml_decisiontree(X_train, X_test, y_train, y_test)
#ml_randomforest(X_train, X_test, y_train, y_test)
#ml_xgboost(X_train, X_test, y_train, y_test)

reduced_stats = [ml_linearreg(X_train, X_test, y_train, y_test), ml_svm(X_train, X_test, y_train, y_test), ml_decisiontree(X_train, X_test, y_train, y_test), ml_randomforest(X_train, X_test, y_train, y_test), ml_xgboost(X_train, X_test, y_train, y_test)]

#compare results from the 3 trials 
ml_results = pd.DataFrame({'All Factors': all_stats, 'Reduced Factors': reduced_stats, 'PCA': PCA})
ml_results.rename(index = {0:'Linear',1:'SVM', 2:'Decision Tree', 3:'Random Forest', 4:'XGBoost'})

