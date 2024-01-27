

# Load Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pgeocode
import plotly.express as px
import plotly.graph_objects as go


# Data Wrangling
# Read the home price csv file from the URL
paths = loads('Untitled-2')
orig_url = paths['orig_url']
file_id = orig_url.split('/')[-2]
data_path= paths['data_path'] + file_id

# Create a dataframe from the data
df = pd.read_csv(data_path, index_col=0)

# Check what I did
print(df.info())
print(df.dtypes())

# Check for duplicate rows, null values
number_of_rows = len(df)
for columns in df:
    non_null_values = df[columns].count()
    amount_of_missing_val_per_column = number_of_rows - non_null_values
    print("There are ", amount_of_missing_val_per_column, "values missing in", columns, ".")



df['duplicate']=df.duplicated()
duplicate_rows_indexes = df.index[df['duplicate']==True].tolist()
df.drop_duplicates() # the good thing is the subset argument that you can specify which value/column duplicates you want to remove


# In[44]:


df.duplicated().sum() # it is adding the booleans


# In[34]:


df[['Currency cut', 'price']] = df.price.str.split("F", expand = True)
df


# In[52]:


df[['price', 'a mark']] = df.price.str.split(".", expand = True)
df.head()


# In[56]:


df['price'] = df['price'].str.replace(',','') # these need to be assigned to parameters
df['price']=df['price'].astype(float)
print(type(df['price'][3]))


# In[75]:


# Keep a new df with only the columns I wants
df_clean = df.drop(labels=['duplicate', 'Currency cut', 'a mark'], axis=1)

# Initialize a 3x3 charts
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

# Flatten the axes array (makes it easier to iterate over)
axes = axes.flatten()

# Loop through each column and plot a histogram
for i, column in enumerate(df_clean.columns):
    
    # Add the histogram
    df_clean[column].hist(ax=axes[i], # Define on which ax we're working on
                    edgecolor='white', # Color of the border
                    color='#69b3a2' # Color of the bins
                   )
    
    # Add title and axis label
    axes[i].set_title(f'{column} distribution') 
    axes[i].set_xlabel(column) 
    axes[i].set_ylabel('Frequency') 

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


# So what I did above is not the best for ALL my columns, because they have different type of values. I could have
# extracted the zip codes or something from there and then get some idea on the address distribution per zip code.


# ---
# The price column is formatted with alphanumeric values. In order to properly do data exploration, we need to treat this column as an integer (number), so let's clean this entries using a Regular Expression (regex) so it only keeps the digits

# In[76]:


df["price"]


# In[77]:


df["price"].sample(n=10).unique()


# In[78]:


# The regex is telling the method to that characters we have given in there are regular expressions
# and not the LITERAL thing we wrote. Without it, it would be looking to replace this group of symbols: [^0-9]
# treating it a normal string.

df['price'] = df['price'].str.replace('[^0-9]', '', regex = True)
df.head()


# ---
# Now we have only numeric values for the price column, but will it be treated as an integer, or will Pandas still interpret it as a string?

# In[79]:


df.info()


# ---
# Even though we stripped the values of non-numerical characters, we still need to convert the data type so it can be interpreted as an integer. This way we can later take advantage of this for plotting and applying methods if needed.

# In[80]:


df['price'] = df['price'].astype(int)
df.info()


# ---
# Another column which should be numerical but it's being treated as an object (string) is the "area_m2" column. In this case it would be beneficial to also strip the "m2" out of the values and only keep the numbers.
# 
# 

# In[15]:


df.head(3)


# In[81]:


df['area_m2'] = df['area_m2'].str.split(' ').str[0]
df['area_m2'] = df['area_m2'].astype(float)
df.head(3)


# ---
# When working with data, many times there will be missing values in some of the samples. This is a normal situation to encounter when analyzing the data and emphasizes the importance of knowing the data.
# 
# There are several approaches to deal with missing values:
# - Substitute the missing values with the mean, median, mode or arbitrary value.
# - Drop the samples with missing data.
# - Impute the missing values using Machine Learning.

# ---
# The column "floors_num" indicates the number of floors that the property has. There are many entries with missing values in this column. We can inspect the data to make a more educated inference of which value to assign to the rows with missing data.
# 
# We can do this by visualizing the counts of each type of property and seeing that most of the properties with missing values are apartments, which most of the time only have one floor.

# In[82]:


print(f"Missing values in 'floors_num': {df['floors_num'].isna().sum()}")


# In[83]:


def get_type_proportion(request):
    if request == "Is Nan":
        conditional = df['floors_num'].isna()
    elif request == "1":
        conditional = df['floors_num'] == 1
    elif request == "More than 1":
        conditional = df['floors_num'] > 1

    s1 = df[conditional]['type'].value_counts()[0:5].sort_index()
    s2 = pd.Series([df[conditional]['type'].value_counts()[5:].sum()], index=["Other"])
    return pd.concat([s1, s2], ignore_index=False)


# In[84]:


fig, ax = plt.subplots(1, 3, figsize=(22,5))

inspect_df = get_type_proportion("1")
ax[0].pie(inspect_df, labels=inspect_df.index, autopct='%1.f%%', startangle=90)
ax[0].set_title("Properties With One Floor", fontsize=18)

inspect_df = get_type_proportion("More than 1")
ax[1].pie(inspect_df, labels=inspect_df.index, autopct='%1.f%%', startangle=90)
ax[1].set_title("Properties With More Than One Floor", fontsize=18)

inspect_df = get_type_proportion("Is Nan")
ax[2].pie(inspect_df, labels=inspect_df.index, autopct='%1.f%%', startangle=90)
ax[2].set_title("Properties With Missing Floor Number", fontsize=18)

fig.show();


# ---
# Looking at the distribution of the types of properties according to the amount of floor levels they have, we can make the following assumption: properties with one floor level are mostly Apartments (73%), so since the majority of properties missing this value are also Apartments (49%), it's more likely that they also have one floor level.

# In[85]:


df['floors_num'] = df['floors_num'].fillna(1).astype(int)
df.head()


# ---
# The "floor" column which represents the level the property is located at also has several missing values. We will infer that no given data means it is a Ground Floor property.

# In[86]:


print(f"Missing values in 'floor': {df['floor'].isna().sum()}")


# In[87]:


# EXERCISE 1
# Fill the missing values of the 'floor' column with "GF"
         # <-- Your code here
    
df['floor'] = df['floor'].fillna('GF').astype(str)
df.tail()


# ---
# For the "last_refurbishment" column, which states the year of the last remodeling, there are also missing values. We will infer that this is because the building has not been refurbished since its construction, and it makes sense to set the date as the year which it was built.

# In[88]:


print(f"Missing values in 'last_refurbishment': {df['last_refurbishment'].isna().sum()}")


# In[89]:


df["year_built"]


# In[90]:


# EXERCISE 2
# Fill the missing values in the column 'last_refurbishment' with the values from the column 'year_built'
       # <--- Your code here
df['last_refurbishment'] = df['last_refurbishment'].fillna(df['year_built']).astype(float)
df.tail()


# ## Basic feature extraction

# ---
# From data we already have, we can create new features. For example, using the area of the property and the price, we can also get the price per square meter.

# In[91]:


df = df[df["area_m2"].notna()].copy()
df["price_sqm"] = df['price'] / df['area_m2']
df["price_sqm"] = df["price_sqm"].astype(int)
print(df.shape)
df.head()


# ---
# We can get a few more features from the address column: like separating the zip code and the city, and even getting the coordinates for each property.

# In[92]:


for address in df['address'].sample(n=20).unique():
  print(address)


# ---
# We see that some addresses only contain the zip code and city, but others also contain the full address. Let's extract these values.

# In[93]:


def extract_zip_city(address):
    if ',' in address:
        zip_and_city = address.split(', ')[1]
        zip_code = zip_and_city.split(' ')[0]
        city = zip_and_city.split(' ')[1]
    else:
        zip_and_city = address
        zip_code = zip_and_city.split(' ')[0]
        city = zip_and_city.split(' ')[1]
    return pd.Series([zip_code, city])

df[['zip_code', 'city']] = df['address'].apply(extract_zip_city)
df


# ---
# Using the library `pgeocode`, which we installed at the beginning of the notebook, we can use the zip code number to get the name of the canton and the coordinates we will later use.

# In[94]:


pgeocode_nomi = pgeocode.Nominatim('ch') # Query geographical location from a city name or a postal code
pgeocode_nomi.query_postal_code("8134")


# In[95]:


def add_canton(zip_code):
    zip_info = pgeocode_nomi.query_postal_code(zip_code)
    return zip_info["state_name"]

df["canton"] = df["zip_code"].apply(add_canton) # Apply a function along an axis of the DataFrame.
df.head(3)


# In[96]:


# EXERCISE 3
def add_coordinates(zip_code):
    zip_info = pgeocode_nomi.query_postal_code(zip_code)
    # Assign two variables called 'latitude' and 'longitude' with the corresponding keys from the 'zip_info' data
    latitude =  zip_info['latitude']
    longitude =  zip_info['longitude']
    return pd.Series([latitude, longitude])

df[["lat", "lon"]] = df["zip_code"].apply(add_coordinates)
df.head(3)


# ---
# We no longer need the full address, so we can drop this column.

# In[97]:


df = df.drop('address', axis=1)


# ---
# Now we have our final data frame which we will use to do analysis, gather insights and create a machine learning model!

# In[98]:


df


# In[102]:


# Make a plot of the price distribution per house type and per canton

df.groupby(by='type')['price_sqm'].mean()

fig = px.scatter(df, x="type", y="price_sqm", color='canton')
fig.show()


# In[111]:


# Find the most affordable apartment per canton

df_lowest_prices = df.groupby(by=['canton', 'type'])['price'].min()
print(df_lowest_prices)


# In[121]:


# Or I could get the indexes of the smallest prices per canton and then with a lambda function get the addresses
# Now it will not work because I have removed the address from the df

#df_addresses_with_lowest_prices = df.groupby(by=['canton', 'type']).agg({'price': lambda x: df.loc[x.idxmin()]['address']})

#Get indexes of lowest price, so I can then call the whole rows from the OG df

df_indexes_of_lowest_prices = df.groupby(by=['canton', 'type']).agg({'price': 'idxmin'})
indexes = df_indexes_of_lowest_prices['price'].tolist()

# Ok, I will not do that now
# or maybe I will
df_lowest_prices = df.loc[indexes]
df_lowest_prices


# # Basic EDA (Exploratory Data Analysis)

# ## Data Distribution

# <img src="https://pbs.twimg.com/media/E5ePcUdVkAEvEX6?format=jpg&name=small" width="250" style="background:none; border:none; box-shadow:none;" />
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/c/cc/Relationship_between_mean_and_median_under_different_skewness.png" width="700" style="background:none; border:none; box-shadow:none;" />
# 
# 

# In[39]:


fig, ax = plt.subplots(1, 2,figsize=(20,5))

sns.histplot(data=df, x='price', kde=True, stat='density', ax=ax[0])
sns.histplot(data=df, x='area_m2', kde=True, stat='density', ax=ax[1])

fig.suptitle('Density of Price and Area', fontsize=18)
fig.show();


# In[40]:


fig = px.histogram(df, x="price",
                   marginal="box",
                   hover_data=df.columns)
fig.update_layout(
    font={"size":17},
    title_text="Price Distribution on Histogram and Boxplot",
    title_x=0.5,
    )

fig.show()


# ---
# **Understanding Box Plots**

# <center><img src="https://miro.medium.com/max/9000/1*2c21SkzJMf3frPXPAR_gZA.png" width="700" style="background:none; border:none; box-shadow:none;" /></center>

# In[41]:


fig = px.box(df, x="type", y="price")
fig.update_layout(
    font={"size":17},
    title_text="Boxplot Distribution Between Property Types",
    title_x=0.5,
    )
fig.update_xaxes(tickangle=-45)
fig.show()


# ## Scatter Plot

# In[42]:


fig = px.scatter(df, x="area_m2", y="price", color='type')

fig.update_layout(
    font={"size":17},
    title_text="Correlation Between Property Area And Its Price",
    title_x=0.5,
)

fig.show()


# In[43]:


# EXERCISE 4
# Plot a scatter plot with the correlation between price and number of roooms

fig = px.scatter(df, x="room_num", y="price", color='type')

fig.update_layout(
    font={"size":17},
    title_text="Correlation Between Property Number Of Rooms And Its Price",
    title_x=0.5,
)

fig.show()


# ## Viewing Variable Correlations in a Heatmap

# In[57]:


df2 = pd.DataFrame()

for column in df:
    s = df[column].dtype
    if s != object:
        df2[column] = df[column]

corr_matrix = df2.corr()
corr_matrix


# In[58]:


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

trimask = np.triu(np.ones_like(corr_matrix, dtype=bool))


fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = corr_matrix.columns,
        y = corr_matrix.index,
        z = np.array(corr_matrix),
        text=trunc(np.array(corr_matrix), decs=2), texttemplate="%{text}",
        colorscale = 'RdBu', ygap=1, xgap=1
    )
)

fig.update_layout(
    title_text="Correlation Heatmap",
    title_x=0.5,
    width=1000,
    height=600,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    yaxis_autorange='reversed'
)

fig.show()


# # More Visualizations

# ## Mapping

# In[59]:


fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    hover_name="price",
    color="canton",
    size="price",
    zoom=7,
    center={"lat":46.8182, "lon":8.2275}
)

fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":0,"l":0,"b":0},
    height=600,
    font={"size":17}
)

fig.show()


# In[60]:


# EXERCISE 5
df_map = df[df['price']<5000000]

fig = px.scatter_mapbox(
    df_map,
    lat="lat",
    lon="lon",
    hover_name="price",
    color='price',   #<--- Make the map show the color scale from the price values
    zoom=7,
    center={"lat":46.8182, "lon":8.2275},
)

fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r":0,"t":0,"l":0,"b":0},
    height=600,
    font={"size":17}
)

fig.show()


# # Intro to Machine Learning

# ## Further Data Cleaning

# In[61]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Distribution Of Price')
ax.boxplot(df['price'])
fig.show();


# In[62]:


df = df[df['price']< 5000000].reset_index(drop=True)
#df = df[~df["year_built"].isna()]
df.drop(['lat', 'lon', 'price_sqm'], axis=1, inplace=True, errors='ignore')
df


# ---
# Machine learning is the process of teaching a computer to learn patterns from data and then to apply those patterns to make preditions on new data. In traditional programming, you write rules to tell the computer exactly what to do. For example, if you want to write a program that converts miles to kilometers, you would write a function that computes the following equation:
# 
# <span>
# <img src="https://drive.google.com/uc?id=1aa50Dd83JwO7x_SOWbNckj2ThBOfVdeb" width="40%"/>
# 
# 
# But in ML, instead of writing the rule, you provide the computer a lot of examples of input data as well as the desired output, say many samples miles to kilometer conversion data. Then let the computer learn the rule itself.
# 
# 
# <img src="https://drive.google.com/uc?id=1sj2IeZGi9RI6VH-ZvFpFdS2fC6e3XO0R" width="40%"/>
# </span>
# 
# But there are many cases where the rules are not that simple. For example, this very dataset of housing prices takes into consideration many variables, and it would be very complicated to write a formula ourselves.
# 
# ML is ideal for these types of problems, where you have lots of data that have complex relationships that would be very difficult for humans to manually create rules for.

# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor


# ---
# Don't worry too much about the following code. There are several technicalities better saved for another time. The only important thing to understand is that we separate our data from the independent (x) and dependent variables (y), since we want the ML algorithm to learn the patterns from the independent variables that give the target output.

# In[ ]:


X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
categorical_transformer = Pipeline(steps=[
                                          ("onehot", OneHotEncoder(handle_unknown="ignore"))
                                          ])
numeric_transformer = Pipeline(steps=[
                                      ("knn_imputer", KNNImputer(n_neighbors=5)),
                                      ("scaler", MinMaxScaler())
                                      ])
preprocessor = ColumnTransformer(transformers=[
                                               ("num", numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features)
                                               ])

model = RandomForestRegressor(n_estimators=1000)

pipeline_model = Pipeline(steps=[
                              ("pre_process", preprocessor),
                              ("model", model)
                              ])

pipeline_model.fit(X_train, y_train)


# In[ ]:


pred = pipeline_model.predict(X_test)

print('MAE', metrics.mean_absolute_error(y_test, pred))
print('R2 Score', metrics.r2_score(y_test, pred))


# ---
# Enter the values of a property you would like to predict its price for:

# In[ ]:


# EXERCISE 6

target_property = {
    'type' : ['Apartment'],
    'room_num' : [2.5],
    'floor' : ["2"],
    'area_m2' : [80],
    'floors_num' : [1],
    'year_built' : [1990],
    'last_refurbishment' : [2002],
    'zip_code' : ["8003"],
    'city' : ["Zürich"],
    'canton' : ["Kanton Zürich"],
}

to_predict = pd.DataFrame(target_property)
to_predict[['area_m2', 'year_built', 'last_refurbishment']] = to_predict[['area_m2', 'year_built', 'last_refurbishment']].astype(float)
to_predict


# In[ ]:


pred = pipeline_model.predict(to_predict)
print(f"The value of the property using the trained machine learning algorithm is of {round(pred[0])} CHF")


# ---
# Congratulations!
# 
# - You learned how to start a Data Science project.
# - You learned how to do data wrangling to clean the data.
# - You learned how to do exploratory data analysis and visualize insights with plots.
# - And you trained a Machine Learning Algorithm that allows user to get a predicted price of a property based on previous data.
# 
# It feels awesome to know all these tools.
# 
# From the Constructor Learning team, we thank you for your participation!
# 

# In[ ]:





# ## Solutions to the exercises:
# 
# 
# Exercise 1:
# ```
# df['floor'] = df['floor'].fillna("GF")
# ```
# 
# Exercise 2:
# ```
# df['last_refurbishment'] = df['last_refurbishment'].fillna(df['year_built'])
# ```
# 
# Exercise 3:
# ```
# latitude = zip_info["latitude"]  #<-- Make this EXERCISE
# longitude = zip_info["longitude"] #<-- Make this EXERCISE
# ```
# 
# Exercise 4:
# ```
# fig = px.scatter(df, x="room_num", y="price", color='type')
# ```
# 
# Exercise 5:
# ```
# fig = px.scatter_mapbox(
#     df_map,
#     lat="lat",
#     lon="lon",
#     hover_name="price",
#     color="price",
#     zoom=7,
#     center={"lat":46.8182, "lon":8.2275},
# )
# ```
