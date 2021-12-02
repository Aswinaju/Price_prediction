import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


matplotlib.rcParams['figure.figsize'] = (20,10)

df1 = pd.read_csv("Bengaluru_House_Data.csv")
df2 = df1.drop(['area_type','availability','society','balcony'],axis = 'columns')
df3 = df2.dropna()

df3['BHK'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
#print(df3['total_sqft'].unique())

#Function to convetr total sqft to float
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


df3[~df3['total_sqft'].apply(is_float)]

#Function to convetr range of total sqft to float
def range_to_float(y):
    tokens = y.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(y)
    except:
        return None

df5 = df3.copy()

df5['total_sqft'] = df3['total_sqft'].apply(range_to_float)

#adding new column called price per sqft

df5['price_per_sqft'] = df5['price']*100000 / df5['total_sqft']
df5['location'] = df5['location'].apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)

#Renaming locations having counts less than to others

lesser_than_10 = location_stats[location_stats <= 10]
df5['location'] = df5.location.apply(lambda x: 'others' if x in lesser_than_10 else x)

#Removing the datas having more number of bathrooms than bedrooms

df6 = df5[~(df5.total_sqft / df5.BHK < 300)]

#Removing Outliers

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft >(m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index= True)
    return df_out

df7 = remove_pps_outliers(df6)


# Scatter plot

def scatter_plot(df,location):
    bhk2 = df[(df.location == location) & (df.BHK == 2)]
    bhk3 = df[(df.location == location) & (df.BHK == 3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',s=50,label = '2 BHK')
    plt.scatter(bhk3.total_sqft, bhk3.price, color='red', s=50, label='3 BHK',marker='*')
    plt.xlabel('total_sqft')
    plt.ylabel('price')
    plt.title(location)
    plt.legend()
    plt.show()

#Removing the 2BHK price more than 3BHK

def remove_bhk_outlier(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std':  np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


df8 = remove_bhk_outlier(df7)

#Removing the houses having more bathroom than bedroom and droping price_per_sqft & size columns
df9 = df8[df8.bath < df8.BHK+2]
df10 = df9.drop(['price_per_sqft','size'],axis = 'columns')

#converting an object to numbers by creating dummies
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dummies.drop('others',axis='columns')],axis='columns')
df12 = df11.drop('location',axis= 'columns')

#creating dependent and independent values and splitting test and train values
x = df12.drop('price',axis= 'columns')
y = df12.price

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=10)

lg = LinearRegression()
lg.fit(x_train.values,y_train.values)


def price_predicting(location,sqft,bath,BHK):
    loc_index = np.where(x.columns == location)[0][0]
    X = np.zeros(len(x.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return lg.predict([X])

print(price_predicting('1st Phase JP Nagar',1000,3,3))