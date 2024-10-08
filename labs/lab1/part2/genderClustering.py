import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
customer_data = pd.read_csv('shopping_data.csv')

# TASK 1: Distinguish between Male and Female Spenders
# Filter data for male and female customers
male_customers = customer_data[customer_data['Genre'] == 'Male']
female_customers = customer_data[customer_data['Genre'] == 'Female']

# Scatter plot of male and female customers based on Spending Score and Annual Income
plt.figure(figsize=(10, 7))
plt.scatter(male_customers['Annual Income (k$)'], male_customers['Spending Score (1-100)'],
            color='blue', label='Male', s=100)
plt.scatter(female_customers['Annual Income (k$)'], female_customers['Spending Score (1-100)'],
            color='pink', label='Female', s=100)
plt.title('Spending Score and Annual Income by Gender')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# TASK 2: Box Plot for Spending Score and Income by Gender
# Box plot for Spending Score by Gender
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Spending Score (1-100)', data=customer_data)
plt.title('Spending Score by Gender')
plt.show()

# Box plot for Annual Income by Gender
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Annual Income (k$)', data=customer_data)
plt.title('Annual Income by Gender')
plt.show()

# TASK 3: Histograms for Spending and Income Distribution by Gender
# Histogram for Spending Score by Gender
plt.figure(figsize=(10, 6))
sns.histplot(male_customers['Spending Score (1-100)'], color='blue', label='Male', kde=True)
sns.histplot(female_customers['Spending Score (1-100)'], color='pink', label='Female', kde=True)
plt.title('Spending Score Distribution by Gender')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Histogram for Annual Income by Gender
plt.figure(figsize=(10, 6))
sns.histplot(male_customers['Annual Income (k$)'], color='blue', label='Male', kde=True)
sns.histplot(female_customers['Annual Income (k$)'], color='pink', label='Female', kde=True)
plt.title('Annual Income Distribution by Gender')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# TASK 4: Bar Plot for Big Spenders by Gender
# Define big spenders as Spending Score > 70
big_spenders_male = len(male_customers[male_customers['Spending Score (1-100)'] > 70])
big_spenders_female = len(female_customers[female_customers['Spending Score (1-100)'] > 70])

# Bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=['Male', 'Female'], y=[big_spenders_male, big_spenders_female], palette=['blue', 'pink'])
plt.title('Number of Big Spenders by Gender')
plt.ylabel('Number of Big Spenders')
plt.show()

# TASK 5: Correlation Analysis Between Income and Spending for Each Gender
# Correlation for male customers
correlation_male = male_customers[['Annual Income (k$)', 'Spending Score (1-100)']].corr()
print('Correlation between Annual Income and Spending Score for males:\n', correlation_male)

# Correlation for female customers
correlation_female = female_customers[['Annual Income (k$)', 'Spending Score (1-100)']].corr()
print('Correlation between Annual Income and Spending Score for females:\n', correlation_female)

# Scatter plot for males
plt.figure(figsize=(10, 6))
plt.scatter(male_customers['Annual Income (k$)'], male_customers['Spending Score (1-100)'], color='blue', label='Male')
plt.title('Correlation between Annual Income and Spending Score (Male)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Scatter plot for females
plt.figure(figsize=(10, 6))
plt.scatter(female_customers['Annual Income (k$)'], female_customers['Spending Score (1-100)'], color='pink', label='Female')
plt.title('Correlation between Annual Income and Spending Score (Female)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# TASK 6: Descriptive Statistics by Gender
# Descriptive statistics for males
print("\nMale Customers Descriptive Statistics:")
print(male_customers[['Annual Income (k$)', 'Spending Score (1-100)']].describe())

# Descriptive statistics for females
print("\nFemale Customers Descriptive Statistics:")
print(female_customers[['Annual Income (k$)', 'Spending Score (1-100)']].describe())

# TASK 7: Clustering Separately for Males and Females
# Clustering for male customers
male_data = male_customers.iloc[:, 3:5].values
male_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
male_pred = male_cluster.fit_predict(male_data)

plt.figure(figsize=(10, 6))
plt.scatter(male_customers['Annual Income (k$)'], male_customers['Spending Score (1-100)'], c=male_pred, cmap='coolwarm', s=100)
plt.title('Male Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Clustering for female customers
female_data = female_customers.iloc[:, 3:5].values
female_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
female_pred = female_cluster.fit_predict(female_data)

plt.figure(figsize=(10, 6))
plt.scatter(female_customers['Annual Income (k$)'], female_customers['Spending Score (1-100)'], c=female_pred, cmap='coolwarm', s=100)
plt.title('Female Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
