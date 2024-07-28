#!/usr/bin/env python
# coding: utf-8

# # Electric Vehicles Market Size Analysis using Python

# In[60]:


# Imported all the libraries which are in used:-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit


# In[3]:


ev_data = pd.read_csv('Electric_Vehicle_Population_Data.csv')


# In[4]:


ev_data.head()


# In[5]:


# This Data based on the Ev population in the United States:-
# Now it's time to clean the dataset:-

ev_data.info()


# In[6]:


ev_data.isnull().sum()


# In[7]:


ev_data = ev_data.dropna()


# In[8]:


ev_data


# In[9]:


# Analyzing the EV Adoption over time by visualizing the number of EVs registered by model year
# So Now created the bar plot:-
sns.set_style("whitegrid")
plt.figure(figsize = (12, 7))
ev_adoption_by_year = ev_data['Model Year'].value_counts().sort_index()
sns.barplot(x = ev_adoption_by_year.index, y = ev_adoption_by_year.values, palette = "rainbow")
plt.title("EV Adoption Over year")
plt.xlabel("Model year")
plt.ylabel("Number of Vechicles Registered")
plt.tight_layout()
plt.show()


# From the above bar chart, it’s clear that EV adoption has been increasing over time, especially noting a significant upward trend starting around 2016. The number of vehicles registered grows modestly up until that point and then begins to rise more rapidly from 2017 onwards. The year 2023 shows a particularly sharp increase in the number of registered EVs, with the bar for 2023 being the highest on the graph, indicating a peak in EV adoption

# In[10]:


# Selecting the Top 10 counties based on EV registrations:-
# Georiphical distribution at country level:-

ev_county_distribution = ev_data['County'].value_counts()
top_counties = ev_county_distribution.head(3).index


# In[11]:


# Filtring the datasets for these top counties:-
top_counties_data = ev_data[ev_data['County'].isin(top_counties)]    # isin(filtring the data)


# In[12]:


# Analyzing the distributions of EVs within the cities of these top Counties:-
ev_city_distribution_top_counties = top_counties_data.groupby(['County', 'City']).size().sort_values(ascending = False).reset_index(name = 'Number of Vehicles')


# In[13]:


# Visualize the top 10 cities across these counties:-
top_cities = ev_city_distribution_top_counties.head(10)


# In[14]:


# Plotting the bargraph:-
plt.figure(figsize = (12, 8))
sns.barplot(x = 'Number of Vehicles', y = 'City', hue = 'County', data = top_cities, palette = 'magma')
plt.title("Top Cities in Top Counties by EV Registrations")
plt.xlabel('Number of Vehicles Registered')
plt.ylabel('City')
plt.tight_layout()
plt.show()


# # The above graph compares the number of electric vehicles registered in various cities within three counties: King, Snohomish, and Pierce. The horizontal bars represent cities, and their length corresponds to the number of vehicles registered, colour-coded by county. Here are the key findings from the above graph:

# In[15]:


# Analyzing the distribution of electric vehicles types:-
ev_type_distribution = ev_data['Electric Vehicle Type'].value_counts()


# In[16]:


# Plotting the BarGraph:-
plt.figure(figsize = (12, 6))
sns.barplot(x = ev_type_distribution.values, y = ev_type_distribution.index, palette = 'rocket')
plt.title('Distribution of Electric Vehicles Type')
plt.xlabel('Number of Vehicles Registered')
plt.ylabel('Electric vehicles Type')
plt.tight_layout()
plt.show()


# # The above graph shows that BEVs are more popular or preferred over PHEVs among the electric vehicles registered in the United States.

# In[17]:


# Analyzing the popularities of EV Manufactures:-
ev_makes_distributions = ev_data['Make'].value_counts().head(10)


# In[18]:


# Now here is the plotting with BarGraph:-
plt.figure(figsize = (12, 7))
sns.barplot(x = ev_makes_distributions.values, y = ev_makes_distributions.index, palette = 'coolwarm_r')
plt.title('Top 10 Popular EV Makes')
plt.xlabel('Number of Vehicles Regitered')
plt.ylabel('Makes')
plt.tight_layout()
plt.show()


# The above chart shows that:
# 
# TESLA leads by a substantial margin with the highest number of vehicles registered.
# NISSAN is the second most popular manufacturer, followed by CHEVROLET, though both have significantly fewer registrations than TESLA.
# FORD, BMW, KIA, TOYOTA, VOLKSWAGEN, JEEP, and HYUNDAI follow in decreasing order of the number of registered vehicles.

# In[19]:


# Selectig the top 3 manufacturing based on the number of vehicles registered:-
top_3_makes = ev_makes_distributions.head(3).index


# In[20]:


# Filtring the datasets for these top manufacture:-
top_makes_data = ev_data[ev_data['Make'].isin(top_3_makes)]


# In[21]:


# analysizing the popularities of EV models within the top manufactures:-
ev_models_distribution_top_makes = top_makes_data.groupby(['Make', 'Model']).size().sort_values(ascending = False).reset_index(name = 'Number of Vehicles')


# In[22]:


# Visualization the top 10 models accross the manufacture for calirity:-
top_models = ev_models_distribution_top_makes.head(10)


# In[32]:


# Plotting the barplot of top 10 models:-
plt.figure(figsize = (12, 7))
sns.barplot(x = 'Number of Vehicles', y = 'Model', hue = 'Make', data = top_models, palette = 'gist_heat')
plt.title('Top Models in Top 10 3 Makes By EV Registration')
plt.xlabel('Number of Vehicles Registred')
plt.ylabel('Model')
plt.tight_layout()
plt.show()


# The above graph shows the distribution of electric vehicle registrations among different models from the top three manufacturers: TESLA, NISSAN, and CHEVROLET. Here are the findings:
# 
# TESLA’s MODEL Y and MODEL 3 are the most registered vehicles, with MODEL Y having the highest number of registrations.
# NISSAN’s LEAF is the third most registered model and the most registered non-TESLA vehicle.
# TESLA’s MODEL S and MODEL X also have a significant number of registrations.
# CHEVROLET’s BOLT EV and VOLT are the next in the ranking with considerable registrations, followed by BOLT EUV.
# NISSAN’s ARIYA and CHEVROLET’s SPARK have the least number of registrations among the models shown.

# The electric range indicates how far an EV can travel on a single charge, and advancements in battery technology have been steadily increasing these ranges over the years. So, let’s look at the distribution of electric ranges in the dataset and identify any notable trends, such as improvements over time or variations between different vehicle types or manufacturers:

# In[46]:


# So, here is Analysize the distribution of electrc range:-
plt.figure(figsize = (12, 6))
sns.histplot(ev_data['Electric Range'], bins = 30, kde = True, color = 'royalblue')
plt.title('Distribution of Electric Vehicles range')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Number of Vehicles')
plt.axvline(ev_data['Electric Range'].mean(), color = 'Red', linestyle = '--', label = f'Mean Range: {ev_data["Electric Range"].mean():.2f} miles')
plt.legend()
plt.show()


# The above graph shows the mean electric range. Key observations from the graph include:
# 
# There is a high frequency of vehicles with a low electric range, with a significant peak occurring just before 50 miles.
# The distribution is skewed to the right, with a long tail extending towards higher ranges, although the number of vehicles with higher ranges is much less frequent.
# The mean electric range for this set of vehicles is marked at approximately 58.84 miles, which is relatively low compared to the highest ranges shown in the graph.
# Despite the presence of electric vehicles with ranges that extend up to around 350 miles, the majority of the vehicles have a range below the mean.

# In[47]:


# Now calculate the average electric range by model year:-
average_range_by_year = ev_data.groupby('Model Year')['Electric Range'].mean().reset_index()


# In[80]:


# Now plotting the lineplot of range by model year:-
plt.figure(figsize = (12, 6))
sns.lineplot(x = 'Model Year', y = 'Electric Range', data = average_range_by_year, marker = 'o', color = 'green')
plt.title('Average Electric Range by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Average Electric Range (miles)')
plt.grid(True)
plt.show('Average Electric Range by Model Year')

plt.savefig('Average Electric Range by Model Year')


# The above graph shows the progression of the average electric range of vehicles from around the year 2000 to 2024. Key findings from the graph:
# 
# There is a general upward trend in the average electric range of EVs over the years, indicating improvements in technology and battery efficiency.
# There is a noticeable peak around the year 2020 when the average range reaches its highest point.
# Following 2020, there’s a significant drop in the average range, which could indicate that data for the following years might be incomplete or reflect the introduction of several lower-range models.
# After the sharp decline, there is a slight recovery in the average range in the most recent year shown on the graph.

# In[51]:


# Let's explore the electric range the top manufactures and models:-
average_range_by_models = top_makes_data.groupby(['Make', 'Model'])['Electric Range'].mean().sort_values(ascending = False).reset_index()


# In[52]:


# The top 10 models highest average electric range:-
top_range_models = average_range_by_models.head(10)


# In[54]:


# Now here, is plotting the barplot so:-
plt.figure(figsize = (12, 7))
barplot = sns.barplot(x = 'Electric Range', y = 'Model', hue = 'Make', data = top_range_models, palette = 'cool')
plt.title('Top 10 Models of Average Electric Range in Top Markets')
plt.xlabel('Average Electric miles (Range)')
plt.ylabel('Model')
plt.show()


# The TESLA ROADSTER has the highest average electric range among the models listed. TESLA’s models (ROADSTER, MODEL S, MODEL X, and MODEL 3) occupy the majority of the top positions, indicating that on average, TESLA’s vehicles have higher electric ranges. The CHEVROLET BOLT EV is an outlier among the CHEVROLET models, having a substantially higher range than the VOLT and S-10 PICKUP from the same maker. NISSAN’s LEAF and CHEVROLET’s SPARK are in the lower half of the chart, suggesting more modest average ranges.

# # Estimated Market Size Analysis of Electric Vehicles in the United States

#  The estimated market size of electric vehicles in the United States. I’ll first count the number of EVs registered every year:

# In[57]:


# Calculate the number of EVs registred in each year:-
ev_registred_counts = ev_data['Model Year'].value_counts().sort_index()


# In[58]:


ev_registred_counts


# The dataset provides the number of electric vehicles registered each year from 1997 through 2024. However, the data for 2024 is incomplete as it only contains the data till March. Here’s a summary of EV registrations for recent years:
# 
# In 2021, there were 19,063 EVs registered.
# In 2022, the number increased to 27708 EVs.
# In 2023, a significant jump to 57,519 EVs was observed.
# For 2024, currently, 7,072 EVs are registered, which suggests partial data.

# We’ll calculate the Compound Annual Growth Rate (CAGR) between a recent year with complete data (2023) and an earlier year to project the 2024 figures. Additionally, using this growth rate, we can estimate the market size for the next five years. Let’s proceed with these calculations:

# In[61]:


# Here is the used Scipy and NupPy Libraries:-
# Filtre the dataset include year with complete data, assuming 2023 is the last complete year:-

filtered_year = ev_registred_counts[ev_registred_counts.index <= 2023]


# In[62]:


# Define a function of expontial grow to fit the data:-
def expontial_growth(x, a, b):
    return a * np.exp(b * x)


# In[64]:


# Prepare the data for curve fitting:-
x_data = filtered_year.index - filtered_year.index.min()
y_data = filtered_year.values


# In[65]:


# fit the data into the exponential growth function:-
params, covariance = curve_fit(expontial_growth, x_data, y_data)


# In[68]:


# Used the fitted function to forecast the number of EVs for 2024 to next for five years:-
forecast_year = np.arange(2024, 2024 + 6) - filtered_year.index.min()
forecasted_values = expontial_growth(forecast_year, * params)


# In[69]:


# Create a distonaries to the display the forecasted values for easier:-
forecasted_evs = dict(zip(forecast_year + filtered_year.index.min(), forecasted_values))


# In[71]:


print(forecasted_evs)


# In[72]:


# now let's plot the estimited market size data:-
# Prepare data for plotting

years = np.arange(filtered_year.index.min(), 2029 + 1)
actual_year = filtered_year.index
forecast_year_full = np.arange(2024, 2029 + 1)


# In[74]:


# Actual and forecasted values:-
actual_values = filtered_year.values
forecasted_values_full = [forecasted_evs[year] for year in forecast_year_full]


# In[78]:


# Plot the Finalized Graph:-
plt.figure(figsize = (12, 8))
plt.plot(actual_year, actual_values, 'bo-', label = 'Actual Registrations')
plt.plot(forecast_year_full, forecasted_values_full, 'ro-', label = 'Forecasted Registrations')
plt.title('Current and Estimited EV Markets')
plt.xlabel('Years')
plt.ylabel('Number of EVs Registrations')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('Current and Estimited EV Markets')


# The green color represented the Actual Market registred and Red id one is Forecaste Market Registrations:- 

# From the above graph, we can see:
# 
# The number of actual EV registrations remained relatively low and stable until around 2010, after which there was a consistent and steep upward trend, suggesting a significant increase in EV adoption.
# The forecasted EV registrations predict an even more dramatic increase in the near future, with the number of registrations expected to rise sharply in the coming years.

# In[ ]:




