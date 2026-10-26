# Data Science and Visualization Lab Manual

---

## Question Set 1
**1.1** Develop an email dataset to conduct a comprehensive exploratory data analysis (EDA). Import the email data into a Pandas DataFrame, visualize key features using appropriate techniques, and extract valuable insights about the data.  
**1.2** Construct a data preprocessing by removing NULL values in a dataset.

### Aim
To create a synthetic email dataset, import it into a Pandas DataFrame to perform Exploratory Data Analysis (EDA) with visualizations, and demonstrate data preprocessing techniques by identifying and removing NULL values from a dataset.

### Procedure
1. **Initialize the Environment:** Open your Python development environment (Jupyter Notebook, Google Colab, or IDE) and import the required libraries: `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`.
2. **Generate Email Dataset:** Create a dictionary representing email metadata containing fields such as `Email_ID`, `Sender_Domain`, `Email_Length`, `Attachments`, `Spam_Score`, and `Label` (Spam/Ham).
3. **Introduce Missing Data:** Intentionally insert `np.nan` (NULL) values into specific rows and columns of the dictionary to simulate real-world noisy data.
4. **Load into DataFrame:** Convert the dictionary into a Pandas DataFrame using `pd.DataFrame()`.
5. **Initial Data Inspection:** Display the first five rows using `.head()`, inspect data types using `.info()`, and generate summary statistics using `.describe()`.
6. **Check for NULL Values:** Use `.isnull().sum()` to count the number of missing values in each column before preprocessing.
7. **Perform Data Preprocessing (Removing NULLs):** Apply the `.dropna()` method to remove rows containing NULL values and store the cleaned data in a new DataFrame. Verify the removal by running `.isnull().sum()` again.
8. **Feature Visualization - Distribution:** Plot a histogram and Kernel Density Estimate (KDE) of `Email_Length` and `Spam_Score` using `seaborn.histplot()` to understand feature distributions.
9. **Feature Visualization - Categorical:** Create a count plot of `Sender_Domain` grouped by `Label` (Spam vs. Ham) and a boxplot comparing `Email_Length` across labels.
10. **Extract Insights & Finalize:** Configure plot titles, axis labels, and legends using Matplotlib, render the visualizations with `plt.show()`, and print key analytical insights extracted from the cleaned dataset.

### Program
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2 & 3: Develop an email dataset with intentional NULL values
np.random.seed(42)
data = {
    'Email_ID': range(1, 21),
    'Sender_Domain': ['gmail.com', 'yahoo.com', 'corp.com', 'spammail.net', 'gmail.com'] * 4,
    'Email_Length': [120, 450, np.nan, 890, 230, 150, 600, 310, np.nan, 780, 
                     110, 520, 340, 910, 205, 180, np.nan, 400, 850, 290],
    'Attachments': [0, 2, 1, 0, 0, 1, 3, np.nan, 0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 0, 1],
    'Spam_Score': [0.1, 0.3, 0.85, 0.95, 0.05, 0.2, 0.15, 0.75, 0.92, 0.88,
                   0.12, 0.25, 0.4, 0.99, 0.08, 0.18, 0.65, 0.35, 0.91, 0.22],
    'Label': ['Ham', 'Ham', 'Spam', 'Spam', 'Ham', 'Ham', 'Ham', 'Spam', 'Spam', 'Spam',
              'Ham', 'Ham', 'Ham', 'Spam', 'Ham', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham']
}

# Step 4: Import into Pandas DataFrame
df = pd.DataFrame(data)
print("--- Original Dataset Head ---")
print(df.head())

# Step 5 & 6: Inspect NULL values
print("\n--- NULL Values Before Preprocessing ---")
print(df.isnull().sum())

# Step 7: Data Preprocessing by removing NULL values
df_cleaned = df.dropna().reset_index(drop=True)
print("\n--- NULL Values After Preprocessing ---")
print(df_cleaned.isnull().sum())
print(f"\nRows before cleaning: {len(df)}, Rows after cleaning: {len(df_cleaned)}")

# Step 8 & 9: Visualize Key Features
plt.figure(figsize=(12, 5))

# Subplot 1: Boxplot of Email Length by Label
plt.subplot(1, 2, 1)
sns.boxplot(x='Label', y='Email_Length', data=df_cleaned, palette='Set2')
plt.title('Email Length Distribution by Label')
plt.xlabel('Email Category')
plt.ylabel('Length (Words)')

# Subplot 2: Count plot of Sender Domain vs Label
plt.subplot(1, 2, 2)
sns.countplot(x='Sender_Domain', hue='Label', data=df_cleaned, palette='pastel')
plt.title('Emails by Sender Domain and Label')
plt.xlabel('Sender Domain')
plt.ylabel('Count')
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

# Step 10: Extract Insights
print("\n--- Summary Insights ---")
print(df_cleaned.groupby('Label')[['Email_Length', 'Spam_Score', 'Attachments']].mean())
```

---

## Question Set 2
**2.1** Utilize NumPy arrays and Pandas DataFrames for efficient data manipulation and analysis, coupled with Matplotlib for creating informative visualizations.  
**2.2** Utilize an Employee satisfaction dataset and apply various EDA and visualization techniques and present an analysis report.

### Aim
To perform efficient data manipulation using NumPy arrays and Pandas DataFrames on an Employee Satisfaction dataset, apply exploratory data analysis (EDA) with Matplotlib visualizations, and generate a structured analysis report.

### Procedure
1. **Import Libraries:** Import `numpy` for array operations, `pandas` for DataFrame manipulation, and `matplotlib.pyplot` for plotting.
2. **Generate NumPy Data Arrays:** Create NumPy arrays for employee attributes such as `Employee_ID`, `Satisfaction_Level` (0 to 1), `Last_Evaluation`, `Number_Projects`, `Average_Monthly_Hours`, `Time_Spend_Company`, and `Left_Company` (0 or 1).
3. **Construct Pandas DataFrame:** Combine the NumPy arrays into a structured Pandas DataFrame.
4. **Data Manipulation with NumPy & Pandas:** Compute a new derived column `Workload_Index` using vectorized NumPy operations (`Average_Monthly_Hours` / `Number_Projects`).
5. **Descriptive Statistics:** Execute `.describe()` and `.info()` to evaluate central tendency, dispersion, and data structure.
6. **Group-Level Aggregation:** Group the dataset by department and turnover status (`Left_Company`) using `.groupby()` to analyze average satisfaction and monthly hours.
7. **Visualization 1 - Satisfaction Histogram:** Use Matplotlib's `plt.hist()` to plot the distribution of employee satisfaction levels.
8. **Visualization 2 - Scatter Plot:** Create a scatter plot comparing `Average_Monthly_Hours` against `Satisfaction_Level`, color-coded by whether the employee left the company.
9. **Visualization 3 - Department Bar Chart:** Plot a bar chart showing average satisfaction levels across different company departments.
10. **Generate Analysis Report:** Print a formatted statistical report summarizing turnover rates, satisfaction correlations, and actionable HR insights.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Utilize NumPy arrays to generate data
np.random.seed(101)
n_employees = 100
emp_id = np.arange(1001, 1001 + n_employees)
satisfaction = np.round(np.random.uniform(0.1, 1.0, n_employees), 2)
evaluation = np.round(np.random.uniform(0.4, 1.0, n_employees), 2)
projects = np.random.randint(2, 7, n_employees)
monthly_hours = np.random.randint(130, 310, n_employees)
tenure = np.random.randint(1, 10, n_employees)
departments = np.random.choice(['Sales', 'IT', 'HR', 'Marketing', 'Finance'], n_employees)
left = np.where((satisfaction < 0.4) | (monthly_hours > 260), 1, 0)

# Step 3: Construct Pandas DataFrame
df_emp = pd.DataFrame({
    'Employee_ID': emp_id,
    'Department': departments,
    'Satisfaction_Level': satisfaction,
    'Last_Evaluation': evaluation,
    'Number_Projects': projects,
    'Average_Monthly_Hours': monthly_hours,
    'Tenure_Years': tenure,
    'Left': left
})

# Step 4: Data Manipulation using NumPy vectorized operation
df_emp['Workload_Index'] = np.round(df_emp['Average_Monthly_Hours'] / df_emp['Number_Projects'], 2)

# Step 5 & 6: EDA & Aggregations
dept_summary = df_emp.groupby('Department')[['Satisfaction_Level', 'Average_Monthly_Hours', 'Left']].mean()

# Step 7, 8 & 9: Visualizations with Matplotlib
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Plot 1: Histogram of Satisfaction Level
axes[0].hist(df_emp['Satisfaction_Level'], bins=10, color='skyblue', edgecolor='black')
axes[0].set_title('Distribution of Satisfaction Level')
axes[0].set_xlabel('Satisfaction Score')
axes[0].set_ylabel('Number of Employees')

# Plot 2: Scatter Plot (Hours vs Satisfaction)
colors = np.where(df_emp['Left'] == 1, 'crimson', 'seagreen')
axes[1].scatter(df_emp['Average_Monthly_Hours'], df_emp['Satisfaction_Level'], c=colors, alpha=0.7)
axes[1].set_title('Monthly Hours vs Satisfaction (Red = Left)')
axes[1].set_xlabel('Average Monthly Hours')
axes[1].set_ylabel('Satisfaction Level')

# Plot 3: Bar Chart by Department
axes[2].bar(dept_summary.index, dept_summary['Satisfaction_Level'], color='mediumpurple', edgecolor='black')
axes[2].set_title('Avg Satisfaction by Department')
axes[2].set_xlabel('Department')
axes[2].set_ylabel('Mean Satisfaction')

plt.tight_layout()
plt.show()

# Step 10: Present Analysis Report
print("================ EMPLOYEE SATISFACTION ANALYSIS REPORT ================")
print(f"Total Employees Analyzed: {len(df_emp)}")
print(f"Overall Attrition Rate: {df_emp['Left'].mean() * 100:.1f}%")
print(f"Mean Satisfaction (Stayed): {df_emp[df_emp['Left']==0]['Satisfaction_Level'].mean():.2f}")
print(f"Mean Satisfaction (Left):   {df_emp[df_emp['Left']==1]['Satisfaction_Level'].mean():.2f}")
print("\n--- Departmental Summary ---")
print(dept_summary.round(2))
print("=======================================================================")
```

---

## Question Set 3
**3.1** Construct a Plot line graph from NumPy array for your own data.  
**3.2** Model a Daily website visitors Time Series Analysis and apply the various visualization techniques.

### Aim
To construct a line graph from user-defined NumPy arrays and model a Daily Website Visitors Time Series dataset using moving averages, trend decomposition, and time-series visualization techniques.

### Procedure
1. **Import Libraries:** Import `numpy`, `pandas`, and `matplotlib.pyplot`.
2. **Create Custom NumPy Arrays:** Define two 1D NumPy arrays representing custom data (e.g., `Days` and `Server_Response_Time_ms`).
3. **Plot Custom Line Graph:** Use `plt.plot()` with custom markers, line styles, and colors to visualize the NumPy array data.
4. **Generate Time Series Dates:** Use `pd.date_range()` to generate a sequence of 90 consecutive daily timestamps for the website visitors dataset.
5. **Simulate Daily Visitors Data:** Combine a linear upward trend, a weekly seasonal sine wave, and random Gaussian noise using NumPy to simulate realistic daily website traffic.
6. **Build Time Series DataFrame:** Create a Pandas DataFrame indexed by `Date` containing the `Daily_Visitors` metric.
7. **Calculate Rolling Statistics:** Compute the 7-day and 14-day simple moving averages (SMA) using `.rolling(window=7).mean()` to smooth out daily fluctuations.
8. **Extract Day-of-Week Patterns:** Create a `Day_of_Week` column using `.dt.day_name()` to analyze weekly traffic seasonality.
9. **Visualize Time Series & Moving Averages:** Plot the raw daily visitors alongside the 7-day and 14-day rolling averages on a timeline.
10. **Visualize Weekly Seasonality:** Create a bar chart of average visitors by day of the week to highlight peak traffic days and display all plots.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Create own data using NumPy arrays
x_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_array = np.array([45, 52, 48, 65, 70, 68, 85, 90, 82, 105])

# Step 4, 5 & 6: Model Daily Website Visitors Time Series Data
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=90, freq='D')
trend = np.linspace(1000, 1800, 90)
seasonality = 250 * np.sin(2 * np.pi * np.arange(90) / 7)
noise = np.random.normal(0, 80, 90)
visitors = (trend + seasonality + noise).astype(int)

ts_df = pd.DataFrame({'Daily_Visitors': visitors}, index=dates)

# Step 7 & 8: Compute Rolling Averages and Day of Week
ts_df['7_Day_MA'] = ts_df['Daily_Visitors'].rolling(window=7).mean()
ts_df['14_Day_MA'] = ts_df['Daily_Visitors'].rolling(window=14).mean()
ts_df['Day_Name'] = ts_df.index.day_name()

# Step 3, 9 & 10: Visualizations
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Plot 1: Line graph from custom NumPy array
axes[0].plot(x_array, y_array, marker='o', linestyle='--', color='teal', linewidth=2)
axes[0].set_title('Custom Data Line Graph from NumPy Array')
axes[0].set_xlabel('Observation Index')
axes[0].set_ylabel('Metric Value')
axes[0].grid(True, alpha=0.3)

# Plot 2: Time Series with Moving Averages
axes[1].plot(ts_df.index, ts_df['Daily_Visitors'], label='Daily Visitors', color='lightgray', alpha=0.8)
axes[1].plot(ts_df.index, ts_df['7_Day_MA'], label='7-Day Moving Avg', color='blue', linewidth=2)
axes[1].plot(ts_df.index, ts_df['14_Day_MA'], label='14-Day Moving Avg', color='darkorange', linewidth=2)
axes[1].set_title('Daily Website Visitors Time Series Analysis')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Visitors Count')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Average Traffic by Day of Week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_avg = ts_df.groupby('Day_Name')['Daily_Visitors'].mean().reindex(day_order)
axes[2].bar(weekly_avg.index, weekly_avg.values, color='coral', edgecolor='black')
axes[2].set_title('Seasonality: Average Website Visitors by Day of Week')
axes[2].set_xlabel('Day of the Week')
axes[2].set_ylabel('Average Visitors')

plt.tight_layout()
plt.show()
```

---

## Question Set 4
**4.1** Construct a Plot bar plot and box plot from NumPy array for your own data.  
**4.2** Develop data preprocessing by renaming columns and omitting NaN values in a user-defined dataset.

### Aim
To generate a bar plot and a box plot directly from custom NumPy arrays, and to implement data preprocessing by renaming columns and omitting `NaN` values in a user-defined Pandas dataset.

### Procedure
1. **Import Libraries:** Import `numpy`, `pandas`, and `matplotlib.pyplot`.
2. **Initialize Custom NumPy Arrays:** Create a categorical array of product labels and a 2D numerical NumPy array representing regional sales figures for each product.
3. **Construct Bar Plot:** Calculate the mean of each category from the NumPy array and plot a bar chart using `plt.bar()`.
4. **Construct Box Plot:** Pass the NumPy array distributions directly into `plt.boxplot()` to visualize quartiles, medians, and outliers.
5. **Define User-Defined Dataset with NaNs:** Create a Pandas DataFrame with poorly named columns (e.g., `col1`, `col2`, `col3`) and insert `np.nan` values in several cells.
6. **Inspect Uncleaned Dataset:** Print the original DataFrame, its column names, and the count of `NaN` values.
7. **Rename Columns:** Use the `.rename(columns={...})` method to assign clear, descriptive names (`Customer_ID`, `Purchase_Amount`, `Loyalty_Points`).
8. **Omit NaN Values:** Apply `.dropna()` to filter out all records containing missing (`NaN`) values.
9. **Reset DataFrame Index:** Use `.reset_index(drop=True)` so the row indices remain sequential after dropping rows.
10. **Verify and Display Results:** Render the bar and box plots side-by-side and print the cleaned, renamed DataFrame to the console.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Create custom NumPy arrays
categories = np.array(['Product A', 'Product B', 'Product C', 'Product D'])
sales_means = np.array([150, 230, 180, 290])

# 2D NumPy array for Box Plot (4 distributions of 30 samples each)
np.random.seed(10)
box_data = np.column_stack([
    np.random.normal(150, 20, 30),
    np.random.normal(230, 35, 30),
    np.random.normal(180, 15, 30),
    np.random.normal(290, 40, 30)
])

# Step 3 & 4: Plot Bar Plot and Box Plot from NumPy arrays
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(categories, sales_means, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'], edgecolor='black')
axes[0].set_title('Bar Plot from NumPy Array')
axes[0].set_xlabel('Products')
axes[0].set_ylabel('Average Units Sold')

axes[1].boxplot(box_data, labels=categories, patch_artist=True)
axes[1].set_title('Box Plot Distribution from NumPy Array')
axes[1].set_xlabel('Products')
axes[1].set_ylabel('Sales Distribution')

plt.tight_layout()
plt.show()

# Step 5 & 6: Develop user-defined dataset with NaN values
raw_data = {
    'c_id': [101, 102, 103, 104, 105, 106],
    'p_amt': [250.5, np.nan, 410.0, 125.0, np.nan, 330.75],
    'l_pts': [25, 40, np.nan, 12, 50, 33]
}
df_raw = pd.DataFrame(raw_data)
print("--- Original User-Defined Dataset ---")
print(df_raw)

# Step 7: Rename columns
df_renamed = df_raw.rename(columns={
    'c_id': 'Customer_ID',
    'p_amt': 'Purchase_Amount',
    'l_pts': 'Loyalty_Points'
})

# Step 8 & 9: Omit NaN values and reset index
df_cleaned = df_renamed.dropna().reset_index(drop=True)

# Step 10: Display Cleaned Dataset
print("\n--- Cleaned Dataset (Renamed Columns & Omitted NaNs) ---")
print(df_cleaned)
```

---

## Question Set 5
**5.1** Construct a Histogram from NumPy arrays for random numbers between 100.  
**5.2** Design EDA on various variable and row filters in R for cleaning data. Apply various plot features in R on sample datasets and visualize.

### Aim
To construct a histogram in Python using NumPy random number arrays between 0 and 100, and to implement Exploratory Data Analysis (EDA), variable/row filtering, and multi-featured visualizations in R.

### Procedure
1. **Python - Import Libraries:** Import `numpy` and `matplotlib.pyplot` in Python.
2. **Python - Generate Random Numbers:** Use `np.random.randint(0, 100, size=500)` to create a NumPy array of 500 random integers between 0 and 100.
3. **Python - Construct Histogram:** Plot the frequency distribution using `plt.hist()` with customized bins, edge colors, and mean reference lines.
4. **R - Initialize Environment:** Open RStudio or an R environment and load a sample dataset (or create a custom dataframe with missing values and multiple variables).
5. **R - Inspect Dataset:** Use `str()`, `summary()`, and `head()` in R to inspect the structure and summary statistics of the dataset.
6. **R - Apply Variable (Column) Filters:** Select specific relevant variables and drop redundant ones using base R subsetting or `subset(df, select = c(...))`.
7. **R - Apply Row Filters for Data Cleaning:** Filter out `NA` values using `na.omit()` and apply logical row conditions (e.g., filtering records where `Score > 50` and `Age >= 18`).
8. **R - Visualize Histogram & Density:** Use R's `hist()` function with custom color palettes and overlay a density curve.
9. **R - Visualize Boxplot & Scatter Plot:** Create a grouped boxplot (`boxplot(Score ~ Category)`) and an enhanced scatter plot (`plot()`) with custom point symbols (`pch`), colors (`col`), and regression lines (`abline()`).
10. **Finalize Outputs:** Display the Python histogram and execute the R script to generate the cleaned R dataframe and multi-panel diagnostic plots.

### Program

#### Part A: Python Program (Histogram from NumPy Array)
```python
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Generate random numbers between 0 and 100
np.random.seed(42)
random_numbers = np.random.randint(0, 100, size=500)

# Step 3: Construct Histogram
plt.figure(figsize=(8, 5))
plt.hist(random_numbers, bins=10, color='royalblue', edgecolor='black', alpha=0.8)
plt.axvline(random_numbers.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {random_numbers.mean():.1f}')
plt.title('Histogram of Random Numbers (0 to 100) from NumPy Array')
plt.xlabel('Number Value Range')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
```

#### Part B: R Program (Variable/Row Filters & Plot Features)
```R
# Step 4: Create a sample dataset in R with noisy/missing data
set.seed(123)
sample_df <- data.frame(
  ID = 1:30,
  Department = sample(c("CS", "ECE", "MECH"), 30, replace = TRUE),
  Age = sample(c(17:25, NA), 30, replace = TRUE),
  Score = round(runif(30, min = 35, max = 99), 1),
  Attendance = round(runif(30, min = 50, max = 100), 1),
  Temp_Redundant_Col = rnorm(30)
)

# Step 5: Initial EDA Inspection
print("--- Original Summary ---")
print(summary(sample_df))

# Step 6: Variable (Column) Filter - Remove redundant column
var_filtered_df <- subset(sample_df, select = c(ID, Department, Age, Score, Attendance))

# Step 7: Row Filter - Clean NA values and filter valid rows (Age >= 18 and Score > 50)
cleaned_df <- na.omit(var_filtered_df)
cleaned_df <- subset(cleaned_df, Age >= 18 & Score > 50)

print("--- Cleaned Dataset Head ---")
print(head(cleaned_df))

# Step 8 & 9: Apply various plot features in R
par(mfrow = c(1, 2)) # 1 row, 2 columns layout

# Plot 1: Scatter plot with regression line
plot(cleaned_df$Attendance, cleaned_df$Score,
     main = "Attendance vs Score",
     xlab = "Attendance (%)", ylab = "Exam Score",
     pch = 19, col = "darkblue", cex = 1.3)
abline(lm(Score ~ Attendance, data = cleaned_df), col = "red", lwd = 2)
grid()

# Plot 2: Grouped Boxplot
boxplot(Score ~ Department, data = cleaned_df,
        main = "Score Distribution by Department",
        xlab = "Department", ylab = "Score",
        col = c("lightblue", "lightgreen", "lightpink"),
        border = "black")
```

---

## Question Set 6
**6.1** Construct a Plot of any two plots for school management dataset.  
**6.2** Build a dataset and apply row filters in R for cleaning data. Apply Scatter plot features in R on sample datasets and visualize.

### Aim
To construct two analytical visualizations for a School Management dataset in Python, and to build a sample dataset in R, clean it using row filters, and visualize relationships using enhanced scatter plot features.

### Procedure
1. **Python - Import Libraries:** Import `pandas`, `numpy`, and `matplotlib.pyplot`.
2. **Python - Create School Management Dataset:** Construct a DataFrame containing student records across grades (`Grade_Level`, `Average_Marks`, `Attendance_Rate`, `Extracurricular_Hours`).
3. **Python - Plot 1 (Grouped Bar Chart):** Aggregate average marks and attendance rate by `Grade_Level` and plot a bar chart comparing performance across grades.
4. **Python - Plot 2 (Pie Chart / Scatter Plot):** Construct a pie chart showing student enrollment distribution across grades.
5. **R - Build Dataset:** In R, construct a dataframe containing variables `Student_ID`, `Study_Hours`, `Test_Score`, and `Group`, including outlier/invalid rows (e.g., negative study hours or `NA` values).
6. **R - Inspect Uncleaned Data:** Run `summary()` and `nrow()` to check the initial dataset state.
7. **R - Apply Row Filters for Cleaning:** Use `complete.cases()` to remove missing rows and logical row indexing (`Study_Hours > 0 & Test_Score <= 100`) to eliminate invalid data entries.
8. **R - Configure Scatter Plot Features:** Use `plot()` in R to graph `Study_Hours` against `Test_Score`, customizing point shapes (`pch`), point sizes (`cex`), axis limits (`xlim`, `ylim`), and color encoding by group.
9. **R - Add Trendline and Annotations:** Fit a linear model using `lm()` and draw the trendline with `abline()`, followed by adding a legend via `legend()`.
10. **Execute and Verify:** Run both scripts to verify the visual outputs and cleaned data summaries.

### Program

#### Part A: Python Program (Two Plots for School Management Dataset)
```python
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Create School Management Dataset
school_data = {
    'Grade_Level': ['Grade 8', 'Grade 9', 'Grade 10', 'Grade 11', 'Grade 12'],
    'Student_Count': [120, 135, 110, 95, 90],
    'Average_Marks': [74.5, 71.2, 78.9, 76.0, 82.4],
    'Attendance_Rate': [91.0, 88.5, 93.2, 89.0, 94.5]
}
df_school = pd.DataFrame(school_data)

# Step 3 & 4: Construct Two Plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Bar Chart of Average Marks by Grade
axes[0].bar(df_school['Grade_Level'], df_school['Average_Marks'], color='cornflowerblue', edgecolor='black')
axes[0].set_title('School Management: Average Marks by Grade')
axes[0].set_xlabel('Grade Level')
axes[0].set_ylabel('Average Marks (%)')
axes[0].set_ylim(0, 100)

# Plot 2: Pie Chart of Student Enrollment Distribution
axes[1].pie(df_school['Student_Count'], labels=df_school['Grade_Level'], autopct='%1.1f%%',
            startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
axes[1].set_title('Student Enrollment Distribution Across Grades')

plt.tight_layout()
plt.show()
```

#### Part B: R Program (Row Filters & Scatter Plot Features)
```R
# Step 5: Build a dataset in R with noisy rows
school_r_df <- data.frame(
  Student_ID = 1:15,
  Study_Hours = c(2.5, 5.0, -1.0, 7.5, NA, 4.0, 8.0, 6.5, 3.0, 9.0, 1.5, 5.5, 10.0, 4.5, 7.0),
  Test_Score = c(55, 72, 40, 88, 65, 68, 92, 81, 60, 95, 48, 75, 150, 70, 85),
  Stream = factor(c("Science", "Commerce", "Science", "Arts", "Science",
                    "Commerce", "Science", "Arts", "Commerce", "Science",
                    "Arts", "Commerce", "Science", "Arts", "Science"))
)

print("--- Raw Dataset ---")
print(school_r_df)

# Step 7: Apply row filters in R for cleaning data
# Remove NAs, negative study hours, and impossible test scores (>100)
cleaned_r_df <- school_r_df[complete.cases(school_r_df), ]
cleaned_r_df <- cleaned_r_df[cleaned_r_df$Study_Hours > 0 & cleaned_r_df$Test_Score <= 100, ]

print("--- Cleaned Dataset ---")
print(cleaned_r_df)

# Step 8 & 9: Apply Scatter plot features in R
colors <- c("crimson", "forestgreen", "royalblue")[cleaned_r_df$Stream]

plot(cleaned_r_df$Study_Hours, cleaned_r_df$Test_Score,
     main = "Student Study Hours vs Test Score (Cleaned Data)",
     xlab = "Daily Study Hours",
     ylab = "Test Score (out of 100)",
     pch = 16, col = colors, cex = 1.5,
     xlim = c(0, 10), ylim = c(40, 100))

# Add linear regression trendline and legend
abline(lm(Test_Score ~ Study_Hours, data = cleaned_r_df), col = "darkgray", lwd = 2, lty = 2)
legend("bottomright", legend = levels(cleaned_r_df$Stream),
       col = c("crimson", "forestgreen", "royalblue"), pch = 16, title = "Stream")
grid()
```

---

## Question Set 7
**7.1** Design a data preprocessing by performing dropping of columns in a dataset.  
**7.2** Build different variable and row filters in R for data cleaning. Apply various plotting features in R to sample datasets and visualize the results.

### Aim
To demonstrate data preprocessing in Python by dropping irrelevant or high-missingness columns from a dataset, and to build variable/row filters and multi-featured visualizations in R.

### Procedure
1. **Python - Import Pandas:** Import `pandas` and `numpy`.
2. **Python - Create Dataset with Redundant Columns:** Build a DataFrame containing useful analytical columns alongside redundant columns (`Temp_Hash`, `Internal_Code`, `Unverified_Notes`).
3. **Python - Inspect Columns:** Print the initial DataFrame columns and shape using `.columns` and `.shape`.
4. **Python - Drop Columns:** Use `df.drop(columns=[...])` to remove the unnecessary columns from the dataset.
5. **Python - Verify Preprocessing:** Print the resulting DataFrame to confirm the target columns were dropped cleanly.
6. **R - Build Sample Dataset:** In R, create a dataframe simulating product inventory with variables `Product_ID`, `Category`, `Price`, `Stock`, `Defective_Rate`, and `Obsolete_Flag`.
7. **R - Variable Filtering:** Filter out the `Obsolete_Flag` column using column indexing or `subset()`.
8. **R - Row Filtering:** Filter out products with missing prices (`!is.na(Price)`), zero stock (`Stock > 0`), or high defective rates (`Defective_Rate < 0.10`).
9. **R - Apply Plotting Features:** Create a 2-panel visualization in R (`par(mfrow=c(1,2))`) featuring a customized barplot of stock levels and a scatter/line plot of Price vs. Stock.
10. **Finalize Execution:** Run both programs and examine the console logs and visual plots.

### Program

#### Part A: Python Program (Dropping Columns in a Dataset)
```python
import pandas as pd

# Step 2: Create dataset with redundant columns
data = {
    'Customer_ID': [1, 2, 3, 4, 5],
    'Name': ['Arun', 'Divya', 'Karthik', 'Meera', 'Vikram'],
    'Temp_Session_Hash': ['x9f8a', 'b7c2d', 'm4n1p', 'q8w3e', 'z5x6c'],
    'Annual_Spend': [45000, 62000, 38000, 91000, 54000],
    'Internal_Audit_Code': ['A01', 'A01', 'A02', 'A01', 'A03'],
    'Membership_Tier': ['Silver', 'Gold', 'Bronze', 'Platinum', 'Gold']
}

df = pd.DataFrame(data)
print("--- Original Dataset (Shape:", df.shape, ") ---")
print(df)

# Step 4: Drop unnecessary columns
columns_to_drop = ['Temp_Session_Hash', 'Internal_Audit_Code']
df_preprocessed = df.drop(columns=columns_to_drop)

# Step 5: Verify result
print("\n--- Dataset After Dropping Columns (Shape:", df_preprocessed.shape, ") ---")
print(df_preprocessed)
```

#### Part B: R Program (Variable/Row Filters & Plotting Features)
```R
# Step 6: Build sample product dataset in R
inventory_df <- data.frame(
  Product_ID = paste0("P", 101:112),
  Category = rep(c("Electronics", "Appliances", "Accessories"), 4),
  Price = c(450, 120, NA, 800, 35, 600, 250, 95, 1100, 45, 310, 50),
  Stock = c(25, 60, 15, 0, 120, 18, 40, 85, 10, 200, 30, 5),
  Defective_Rate = c(0.02, 0.05, 0.01, 0.03, 0.15, 0.04, 0.02, 0.06, 0.01, 0.08, 0.12, 0.03),
  Obsolete_Flag = rep("N", 12)
)

# Step 7: Variable Filter (Drop 'Obsolete_Flag' and 'Defective_Rate' after filtering)
# Step 8: Row Filter (Remove NA Price, Stock > 0, and Defective_Rate < 0.10)
clean_inv <- subset(inventory_df,
                    !is.na(Price) & Stock > 0 & Defective_Rate < 0.10,
                    select = c(Product_ID, Category, Price, Stock))

print("--- Cleaned R Dataset ---")
print(clean_inv)

# Step 9: Apply various plotting features in R
par(mfrow = c(1, 2))

# Plot 1: Barplot of Stock by Product
barplot(clean_inv$Stock, names.arg = clean_inv$Product_ID,
        col = "steelblue", border = "white",
        main = "Available Stock per Product",
        xlab = "Product ID", ylab = "Units in Stock", las = 2)

# Plot 2: Price vs Stock plot with custom features
plot(clean_inv$Price, clean_inv$Stock,
     type = "p", pch = 17, col = "darkorange", cex = 1.5,
     main = "Product Price vs Stock Level",
     xlab = "Price ($)", ylab = "Stock Units")
text(clean_inv$Price, clean_inv$Stock, labels = clean_inv$Product_ID, pos = 3, cex = 0.7)
grid()
```

---

## Question Set 8
**8.1** Develop the data cleaning procedures and analyze the dataset using scatter plot.  
**8.2** Develop a daily climate Time Series Analysis and apply the various visualization techniques.

### Aim
To implement comprehensive data cleaning procedures (handling missing values, duplicates, and outliers) followed by scatter plot analysis, and to perform a Daily Climate Time Series Analysis using multi-panel visualizations.

### Procedure
1. **Import Libraries:** Import `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`.
2. **Create Noisy Dataset:** Generate a dataset containing missing values (`NaN`), duplicate records, and extreme numerical outliers.
3. **Clean Duplicates & Missing Values:** Remove exact duplicates using `.drop_duplicates()` and impute missing numerical values using the median (`.fillna(df.median())`).
4. **Filter Outliers:** Use the Interquartile Range (IQR) or Z-score threshold to filter out extreme anomalies.
5. **Scatter Plot Analysis:** Plot the cleaned variables on a scatter plot with a regression trendline to analyze correlation.
6. **Generate Daily Climate Time Series:** Create a 365-day date index using `pd.date_range()` and simulate `Temperature_C`, `Humidity_Pct`, and `Wind_Speed_kmh`.
7. **Compute Monthly Resampling:** Use `.resample('ME').mean()` to aggregate daily climate observations into monthly averages.
8. **Compute Rolling Temperature Average:** Calculate a 30-day rolling mean for temperature to observe seasonal shifts.
9. **Visualize Climate Time Series:** Plot daily temperatures with the 30-day moving average and monthly resampled bar/line charts.
10. **Display Findings:** Render all plots clearly with gridlines, legends, and titles.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PART 1: Data Cleaning & Scatter Plot Analysis ---
# Step 2: Create noisy dataset
raw_df = pd.DataFrame({
    'Engine_Size_L': [1.5, 2.0, 2.5, 3.0, np.nan, 2.0, 3.5, 4.0, 1.8, 2.2, 15.0], # 15.0 is an outlier
    'Fuel_Consumption_L100km': [6.5, 8.2, 9.5, 11.0, 7.8, 8.2, 12.5, 14.0, 7.2, np.nan, 45.0]
})

# Step 3 & 4: Data Cleaning Procedures (Duplicates, Missing Values, Outliers)
clean_df = raw_df.drop_duplicates().copy()
clean_df['Engine_Size_L'] = clean_df['Engine_Size_L'].fillna(clean_df['Engine_Size_L'].median())
clean_df['Fuel_Consumption_L100km'] = clean_df['Fuel_Consumption_L100km'].fillna(clean_df['Fuel_Consumption_L100km'].median())

# Remove extreme outliers (Engine Size < 8.0 L)
clean_df = clean_df[clean_df['Engine_Size_L'] < 8.0].reset_index(drop=True)
print("--- Cleaned Dataset ---")
print(clean_df)

# --- PART 2: Daily Climate Time Series Analysis ---
# Step 6: Generate 1 year of daily climate data
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
day_of_year = np.arange(365)

# Simulate seasonal temperature and humidity
temperature = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2.2, 365)
humidity = 65 - 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 5, 365)

climate_df = pd.DataFrame({
    'Temperature_C': temperature,
    'Humidity_Pct': np.clip(humidity, 20, 100)
}, index=dates)

# Step 7 & 8: Rolling mean and Monthly Resampling
climate_df['Temp_30D_MA'] = climate_df['Temperature_C'].rolling(window=30).mean()
monthly_climate = climate_df.resample('ME').mean()

# Step 5 & 9: Visualizations
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Plot 1: Scatter plot of Cleaned Data
sns.regplot(x='Engine_Size_L', y='Fuel_Consumption_L100km', data=clean_df, ax=axes[0],
            color='teal', scatter_kws={'s': 60})
axes[0].set_title('Scatter Plot: Engine Size vs Fuel Consumption')
axes[0].set_xlabel('Engine Size (Liters)')
axes[0].set_ylabel('Fuel Consumption (L/100km)')

# Plot 2: Daily Temperature Time Series
axes[1].plot(climate_df.index, climate_df['Temperature_C'], color='lightsalmon', alpha=0.6, label='Daily Temp')
axes[1].plot(climate_df.index, climate_df['Temp_30D_MA'], color='darkred', linewidth=2, label='30-Day Moving Avg')
axes[1].set_title('Daily Climate Time Series: Temperature')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Temperature (°C)')
axes[1].legend()

# Plot 3: Monthly Average Temperature vs Humidity
axes[2].plot(monthly_climate.index.strftime('%b'), monthly_climate['Humidity_Pct'],
             marker='s', color='royalblue', linewidth=2, label='Avg Humidity (%)')
axes[2].plot(monthly_climate.index.strftime('%b'), monthly_climate['Temperature_C'],
             marker='o', color='darkorange', linewidth=2, label='Avg Temp (°C)')
axes[2].set_title('Monthly Resampled Climate Trends')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Value')
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## Question Set 9
**9.1** Develop various variable and row filters in R for cleaning data. Apply various plot features in R on sample datasets and visualize.  
**9.2** Develop an Education and student performance dataset and apply various EDA and visualization techniques and present an analysis report.

### Aim
To implement variable and row filtering along with custom plotting in R, and to construct an Education and Student Performance dataset in Python to perform Exploratory Data Analysis (EDA), visualizations, and report generation.

### Procedure
1. **R - Build Sample Dataset:** Create an R dataframe containing employee or student records with missing values and redundant columns.
2. **R - Apply Variable & Row Filters:** Subset the columns to retain key attributes and filter rows using `subset()` to remove `NA`s and out-of-range observations.
3. **R - Visualize with Custom Plot Features:** Plot a histogram and a grouped boxplot in R with custom titles, colors, and reference lines.
4. **Python - Import Libraries:** Import `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`.
5. **Python - Create Student Performance Dataset:** Generate a dataset with `Student_ID`, `Gender`, `Parental_Education`, `Study_Hours_Per_Week`, `Math_Score`, `Reading_Score`, and `Writing_Score`.
6. **Python - Feature Engineering:** Compute `Total_Average_Score` across the three subjects and assign a `Pass_Status` label (`Pass` if average $\ge 50$).
7. **Python - Perform Statistical EDA:** Calculate correlation matrices and group-wise performance averages by `Parental_Education`.
8. **Python - Visualization 1 (Correlation Heatmap):** Use `sns.heatmap()` to visualize correlations between study hours and subject scores.
9. **Python - Visualization 2 (Violin/Box Plot):** Compare `Total_Average_Score` across different levels of `Parental_Education`.
10. **Python - Present Analysis Report:** Print a comprehensive summary report detailing pass percentages, top correlations, and group performance metrics.

### Program

#### Part A: R Program (Variable/Row Filters & Plot Features)
```R
# Step 1: Create sample dataset in R
set.seed(99)
raw_r_data <- data.frame(
  ID = 1:25,
  Group = sample(c("Alpha", "Beta", "Gamma"), 25, replace = TRUE),
  Metric_A = c(runif(22, 10, 100), NA, -5, 120),
  Metric_B = rnorm(25, mean = 50, sd = 12),
  Unused_Col = rep("TEMP", 25)
)

# Step 2: Variable and Row Filters in R
cleaned_r <- subset(raw_r_data,
                    !is.na(Metric_A) & Metric_A >= 0 & Metric_A <= 100,
                    select = c(ID, Group, Metric_A, Metric_B))

print("--- Cleaned R Data Summary ---")
print(summary(cleaned_r))

# Step 3: Visualize in R
par(mfrow = c(1, 2))
hist(cleaned_r$Metric_A, col = "lightseagreen", border = "white",
     main = "Distribution of Metric A", xlab = "Metric A Value")
abline(v = mean(cleaned_r$Metric_A), col = "red", lwd = 2, lty = 2)

boxplot(Metric_B ~ Group, data = cleaned_r, col = c("gold", "tomato", "orchid"),
        main = "Metric B Across Groups", xlab = "Group", ylab = "Metric B")
```

#### Part B: Python Program (Education & Student Performance EDA and Report)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 5: Develop Education & Student Performance Dataset
np.random.seed(42)
n = 120
study_hours = np.random.normal(15, 5, n).clip(2, 30)
parent_edu = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.35, 0.35, 0.2, 0.1])

math_score = (40 + 1.8 * study_hours + np.random.normal(0, 8, n)).clip(0, 100)
reading_score = (45 + 1.6 * study_hours + np.random.normal(0, 7, n)).clip(0, 100)
writing_score = (42 + 1.7 * study_hours + np.random.normal(0, 7.5, n)).clip(0, 100)

df_edu = pd.DataFrame({
    'Student_ID': range(1, n + 1),
    'Parental_Education': parent_edu,
    'Study_Hours_Weekly': np.round(study_hours, 1),
    'Math_Score': np.round(math_score, 1),
    'Reading_Score': np.round(reading_score, 1),
    'Writing_Score': np.round(writing_score, 1)
})

# Step 6: Feature Engineering
df_edu['Average_Score'] = df_edu[['Math_Score', 'Reading_Score', 'Writing_Score']].mean(axis=1).round(1)

# Step 8 & 9: Visualizations
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Correlation Heatmap
corr_matrix = df_edu[['Study_Hours_Weekly', 'Math_Score', 'Reading_Score', 'Writing_Score', 'Average_Score']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', ax=axes[0])
axes[0].set_title('Student Performance Correlation Matrix')

# Plot 2: Boxplot by Parental Education
sns.boxplot(x='Parental_Education', y='Average_Score', data=df_edu,
            order=['High School', 'Bachelor', 'Master', 'PhD'], palette='Set3', ax=axes[1])
axes[1].set_title('Average Score by Parental Education Level')
axes[1].set_xlabel('Parental Education')
axes[1].set_ylabel('Average Score')

plt.tight_layout()
plt.show()

# Step 10: Present Analysis Report
print("======== STUDENT PERFORMANCE EDA REPORT ========")
print(f"Total Students Evaluated: {len(df_edu)}")
print(f"Overall Mean Score: {df_edu['Average_Score'].mean():.2f}")
print(f"Correlation (Study Hours vs Avg Score): {corr_matrix.loc['Study_Hours_Weekly', 'Average_Score']:.2f}")
print("\n--- Mean Scores by Parental Education ---")
print(df_edu.groupby('Parental_Education')[['Math_Score', 'Reading_Score', 'Writing_Score', 'Average_Score']].mean().round(2))
print("================================================")
```

---

## Question Set 10
**10.1** Construct a data preprocessing by renaming columns and skipping NaN in user-defined dataset.  
**10.2** Build data preprocessing by eliminating NULL values in a dataset.

### Aim
To perform end-to-end data preprocessing on user-defined datasets by renaming cryptic column headers, skipping `NaN` values during calculations and row filtering, and eliminating `NULL` (`None`/`np.nan`) values using multiple strategies.

### Procedure
1. **Import Libraries:** Import `pandas` and `numpy`.
2. **Create First User-Defined Dataset:** Define a DataFrame with cryptic column headers (`usr_id`, `tr_val`, `rt_scr`) and embedded `np.nan` values.
3. **Rename Columns:** Use `.rename(columns=...)` to transform cryptic column names into readable business labels (`User_ID`, `Transaction_Value`, `Rating_Score`).
4. **Skip NaNs in Aggregations & Rows:** Demonstrate `skipna=True` in Pandas statistical methods and remove rows with `NaN` using `.dropna()`.
5. **Create Second Dataset with NULLs:** Construct another dataset containing Python `None` and `np.nan` NULL values across multiple attributes.
6. **Inspect NULL Distribution:** Use `.isnull().sum()` and `.info()` to identify exact locations of NULL entries.
7. **Eliminate NULL Rows:** Use `.dropna(axis=0, how='any')` to eliminate any record containing a NULL value.
8. **Eliminate NULL Columns (Alternative Demo):** Demonstrate `.dropna(axis=1, thresh=...)` to show column-wise NULL elimination when needed.
9. **Reset Indices:** Apply `.reset_index(drop=True)` on both cleaned DataFrames for clean sequential indexing.
10. **Compare & Verify:** Print before-and-after tables and null counts to verify complete preprocessing.

### Program
```python
import pandas as pd
import numpy as np

# --- PART 1: Renaming Columns and Skipping NaN ---
# Step 2: User-defined dataset with cryptic column names and NaNs
user_data = {
    'usr_id': [1001, 1002, 1003, 1004, 1005],
    'tr_val': [1250.0, np.nan, 3400.5, 890.0, np.nan],
    'rt_scr': [4.5, 3.8, np.nan, 4.9, 4.2]
}
df1 = pd.DataFrame(user_data)
print("--- Dataset 1: Before Preprocessing ---")
print(df1)

# Step 3: Rename columns
df1_renamed = df1.rename(columns={
    'usr_id': 'User_ID',
    'tr_val': 'Transaction_Value',
    'rt_scr': 'Rating_Score'
})

# Step 4: Skipping NaN during aggregation and omitting NaN rows
mean_tx = df1_renamed['Transaction_Value'].mean(skipna=True)
print(f"\nMean Transaction Value (skipping NaN): {mean_tx:.2f}")

df1_clean = df1_renamed.dropna().reset_index(drop=True)
print("\n--- Dataset 1: After Renaming & Omitting NaN Rows ---")
print(df1_clean)

# --- PART 2: Eliminating NULL Values in a Dataset ---
# Step 5: Create second dataset with NULL (None / np.nan) values
null_data = {
    'Order_ID': ['ORD1', 'ORD2', 'ORD3', 'ORD4', 'ORD5', 'ORD6'],
    'Client_Name': ['Alpha Corp', None, 'Beta LLC', 'Gamma Inc', 'Delta Co', None],
    'Order_Qty': [50, 120, np.nan, 85, 200, 150],
    'Unit_Price': [19.99, 24.50, 15.00, None, 32.00, 21.00]
}
df2 = pd.DataFrame(null_data)

# Step 6: Inspect NULL counts
print("\n--- Dataset 2: NULL Count Before Elimination ---")
print(df2.isnull().sum())

# Step 7 & 9: Eliminate NULL values and reset index
df2_eliminated = df2.dropna(axis=0, how='any').reset_index(drop=True)

# Step 10: Verify results
print("\n--- Dataset 2: After Eliminating All NULL Values ---")
print(df2_eliminated)
print("\nRemaining NULL count:", df2_eliminated.isnull().sum().sum())
```

---

## Question Set 11
**11.1** Design House property sales Time Series Analysis and apply the various visualization techniques.  
**11.2** Construct a Healthcare and medical dataset and apply various EDA and visualization techniques and present an analysis report.

### Aim
To design a House Property Sales Time Series Analysis to visualize pricing trends and sales volume over time, and to construct a Healthcare and Medical dataset to perform Exploratory Data Analysis (EDA) and generate a medical insights report.

### Procedure
1. **Import Libraries:** Import `numpy`, `pandas`, `matplotlib.pyplot`, and `seaborn`.
2. **Generate Property Sales Time Series:** Create a monthly date range over 4 years (48 months) and simulate `Median_House_Price`, `Properties_Sold`, and `Property_Type`.
3. **Compute Time Series Metrics:** Calculate a 6-month rolling average for house prices and year-over-year trends.
4. **Visualize Property Time Series:** Plot a dual-chart showing the house price trend with a 6-month moving average and a bar chart of monthly sales volume.
5. **Construct Healthcare & Medical Dataset:** Build a DataFrame of patient records including `Patient_ID`, `Age`, `BMI`, `Blood_Pressure`, `Cholesterol`, `Glucose_Level`, and `Diagnosis` (`Healthy`, `Pre-Risk`, `High-Risk`).
6. **Perform Healthcare EDA:** Inspect summary statistics and group patient vitals by `Diagnosis` category.
7. **Visualize Healthcare Distribution:** Create a boxplot or violin plot comparing `BMI` and `Blood_Pressure` across diagnosis groups.
8. **Visualize Age vs Blood Pressure:** Plot a scatter plot of `Age` vs `Blood_Pressure` colored by `Diagnosis`.
9. **Format Subplots:** Organize both the Property Time Series and Healthcare EDA visualizations into clean figures.
10. **Present Medical Analysis Report:** Output a structured summary report highlighting risk group averages and key clinical correlations.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PART 1: House Property Sales Time Series Analysis ---
np.random.seed(42)
months = pd.date_range(start='2022-01-01', periods=48, freq='ME')
trend_price = np.linspace(250000, 340000, 48)
seasonal_price = 12000 * np.sin(2 * np.pi * np.arange(48) / 12)
house_prices = trend_price + seasonal_price + np.random.normal(0, 5000, 48)
units_sold = (120 + 30 * np.sin(2 * np.pi * np.arange(48) / 12) + np.random.randint(-15, 15, 48)).astype(int)

df_property = pd.DataFrame({
    'Median_Price_USD': house_prices,
    'Units_Sold': units_sold
}, index=months)
df_property['6M_Price_MA'] = df_property['Median_Price_USD'].rolling(6).mean()

# --- PART 2: Healthcare and Medical Dataset ---
n_patients = 150
age = np.random.randint(22, 80, n_patients)
bmi = np.round(np.random.normal(26.5, 4.5, n_patients), 1)
bp_systolic = np.round(100 + 0.5 * age + 0.8 * bmi + np.random.normal(0, 8, n_patients), 0)
cholesterol = np.round(150 + 0.6 * age + np.random.normal(0, 20, n_patients), 0)

diagnosis = np.where((bp_systolic > 145) | (cholesterol > 220), 'High-Risk',
            np.where((bp_systolic > 130) | (bmi > 28), 'Pre-Risk', 'Healthy'))

df_health = pd.DataFrame({
    'Patient_ID': range(1, n_patients + 1),
    'Age': age,
    'BMI': bmi,
    'Blood_Pressure': bp_systolic,
    'Cholesterol': cholesterol,
    'Diagnosis': diagnosis
})

# --- Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Property Price Time Series
axes[0, 0].plot(df_property.index, df_property['Median_Price_USD'], label='Monthly Median Price', color='slategray', alpha=0.7)
axes[0, 0].plot(df_property.index, df_property['6M_Price_MA'], label='6-Month Moving Avg', color='navy', linewidth=2.5)
axes[0, 0].set_title('House Property Sales: Median Price Trend')
axes[0, 0].set_ylabel('Price (USD)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Property Units Sold Time Series
axes[0, 1].bar(df_property.index, df_property['Units_Sold'], width=20, color='teal', alpha=0.8)
axes[0, 1].set_title('Monthly House Properties Sold Volume')
axes[0, 1].set_ylabel('Units Sold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Healthcare Age vs Blood Pressure
sns.scatterplot(x='Age', y='Blood_Pressure', hue='Diagnosis', data=df_health,
                palette={'Healthy': 'green', 'Pre-Risk': 'orange', 'High-Risk': 'red'}, s=70, ax=axes[1, 0])
axes[1, 0].set_title('Healthcare EDA: Age vs Systolic Blood Pressure')

# Plot 4: Healthcare BMI & Cholesterol by Diagnosis
sns.boxplot(x='Diagnosis', y='Cholesterol', data=df_health,
            order=['Healthy', 'Pre-Risk', 'High-Risk'], palette='Pastel1', ax=axes[1, 1])
axes[1, 1].set_title('Cholesterol Distribution Across Diagnostic Groups')

plt.tight_layout()
plt.show()

# Step 10: Healthcare Analysis Report
print("================ HEALTHCARE & MEDICAL EDA REPORT ================")
print(df_health['Diagnosis'].value_counts())
print("\n--- Mean Clinical Vitals by Diagnosis Group ---")
print(df_health.groupby('Diagnosis')[['Age', 'BMI', 'Blood_Pressure', 'Cholesterol']].mean().round(2))
print("=================================================================")
```

---

## Question Set 12
**12.1** Build a Stock exchange Time Series Analysis and apply the various visualization techniques.  
**12.2** Build cartographic visualization for multiple datasets involving various countries of the world, states and districts in India, etc.

### Aim
To build a Stock Exchange Time Series Analysis featuring moving averages, daily returns, and volatility plots, and to construct cartographic visualizations representing global country metrics as well as Indian state and district datasets.

### Procedure
1. **Import Libraries:** Import `numpy`, `pandas`, `matplotlib.pyplot`, and `plotly.express` (for cartographic map projections).
2. **Simulate Stock Exchange Data:** Create a business-day date range (`freq='B'`) for 200 trading days and simulate stock `Close_Price` and `Trading_Volume` using a geometric random walk.
3. **Calculate Financial Indicators:** Compute the 20-day Simple Moving Average (`SMA_20`), 50-day Simple Moving Average (`SMA_50`), and `Daily_Return_Pct` using `.pct_change()`.
4. **Visualize Stock Time Series:** Plot the stock closing price with SMA overlays and a histogram/line plot of daily percentage returns.
5. **Create World Countries Dataset:** Construct a DataFrame containing ISO country codes (`ISO_Code`), `Country`, and `GDP_Growth_Rate`.
6. **Create India States & Districts Dataset:** Build a dataset containing major Indian States/Districts with their geographic coordinates (`Latitude`, `Longitude`), `State`, `District`, and `Literacy_Rate` / `Industrial_Index`.
7. **Build Global Cartographic Visualization:** Use `plotly.express.choropleth` to map global country data on a world map projection.
8. **Build India States/Districts Cartographic Plot:** Plot the Indian states and districts using geographical coordinates (`scatter_geo` and Matplotlib spatial scatter plot bounded to India's lat/lon coordinates) to ensure both interactive HTML and static plot rendering.
9. **Customize Map Layouts:** Add colorbars, hover labels, and geographic scope settings.
10. **Render Visualizations:** Display the stock market charts and cartographic maps.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# --- PART 1: Stock Exchange Time Series Analysis ---
np.random.seed(42)
trading_days = pd.date_range(start='2025-01-01', periods=200, freq='B')
daily_returns = np.random.normal(0.0008, 0.015, 200)
close_prices = 150 * np.cumprod(1 + daily_returns)

df_stock = pd.DataFrame({'Close_Price': close_prices}, index=trading_days)
df_stock['SMA_20'] = df_stock['Close_Price'].rolling(window=20).mean()
df_stock['SMA_50'] = df_stock['Close_Price'].rolling(window=50).mean()
df_stock['Daily_Return_Pct'] = df_stock['Close_Price'].pct_change() * 100

# --- PART 2: Datasets for Cartographic Visualization ---
# 1. World Countries Dataset
df_world = pd.DataFrame({
    'Country': ['India', 'United States', 'China', 'Germany', 'Brazil', 'Australia', 'Japan', 'United Kingdom'],
    'ISO_Code': ['IND', 'USA', 'CHN', 'DEU', 'BRA', 'AUS', 'JPN', 'GBR'],
    'GDP_Growth_Pct': [6.8, 2.5, 4.9, 1.2, 2.9, 2.1, 1.5, 1.8]
})

# 2. India States & Districts Dataset (with Lat/Lon coordinates)
df_india = pd.DataFrame({
    'State': ['Tamil Nadu', 'Tamil Nadu', 'Maharashtra', 'Karnataka', 'Delhi', 'West Bengal', 'Gujarat', 'Kerala'],
    'District': ['Coimbatore', 'Chennai', 'Mumbai', 'Bengaluru', 'New Delhi', 'Kolkata', 'Ahmedabad', 'Ernakulam'],
    'Latitude': [11.0168, 13.0827, 19.0760, 12.9716, 28.6139, 22.5726, 23.0225, 9.9816],
    'Longitude': [76.9558, 80.2707, 72.8777, 77.5946, 77.2090, 88.3639, 72.5714, 76.2999],
    'Development_Index': [84.5, 88.2, 89.0, 87.5, 86.0, 79.4, 83.1, 91.2]
})

# --- Matplotlib Visualizations (Stock + India District Cartography) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Stock Price & Moving Averages
axes[0].plot(df_stock.index, df_stock['Close_Price'], label='Close Price', color='black', alpha=0.8)
axes[0].plot(df_stock.index, df_stock['SMA_20'], label='20-Day SMA', color='blue', linestyle='--')
axes[0].plot(df_stock.index, df_stock['SMA_50'], label='50-Day SMA', color='red', linestyle='-')
axes[0].set_title('Stock Exchange Price & Moving Averages')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Stock Price ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Stock Daily Returns Distribution
axes[1].hist(df_stock['Daily_Return_Pct'].dropna(), bins=25, color='seagreen', edgecolor='black')
axes[1].set_title('Stock Daily Returns Distribution (%)')
axes[1].set_xlabel('Daily Return (%)')
axes[1].set_ylabel('Frequency')

# Plot 3: Cartographic Spatial Plot of Indian States/Districts
scatter = axes[2].scatter(df_india['Longitude'], df_india['Latitude'],
                          c=df_india['Development_Index'], cmap='viridis',
                          s=df_india['Development_Index']*2.5, edgecolors='black')
for _, row in df_india.iterrows():
    axes[2].annotate(f"{row['District']} ({row['State']})",
                     (row['Longitude'] + 0.3, row['Latitude'] + 0.2), fontsize=8)
axes[2].set_xlim(68, 92)
axes[2].set_ylim(7, 33)
axes[2].set_title('Cartographic Map: India States & Districts')
axes[2].set_xlabel('Longitude (°E)')
axes[2].set_ylabel('Latitude (°N)')
axes[2].grid(True, linestyle=':', alpha=0.6)
plt.colorbar(scatter, ax=axes[2], label='Development Index')

plt.tight_layout()
plt.show()

# Interactive World Choropleth Map using Plotly
fig_world = px.choropleth(
    df_world, locations='ISO_Code', color='GDP_Growth_Pct',
    hover_name='Country', color_continuous_scale='Plasma',
    title='Global Cartographic Visualization: GDP Growth Rate by Country'
)
fig_world.show()
```

---

## Question Set 13
**13.1** Utilize Data analysis and representation on a Map using various Map datasets with Mouse Rollover effect, user interaction, etc.  
**13.2** Build a comprehensive time series analysis, utilizing appropriate visualization methods to explore trends, patterns, and anomalies within the data.

### Aim
To represent spatial data on an interactive map featuring mouse rollover tooltips and zoom/pan user interactions, and to build a comprehensive time series analysis capable of identifying trends, seasonal patterns, and statistical anomalies.

### Procedure
1. **Import Libraries:** Import `numpy`, `pandas`, `matplotlib.pyplot`, and `plotly.express`.
2. **Prepare Interactive Map Dataset:** Create a spatial DataFrame of major global tech hubs containing `City`, `Country`, `Latitude`, `Longitude`, `Tech_Startups`, and `Funding_Billion_USD`.
3. **Configure Mouse Rollover & User Interaction:** Use `plotly.express.scatter_geo` with `hover_name` and `hover_data` parameters to display rich tooltips on mouse rollover, and enable interactive zooming and projection switching.
4. **Simulate Comprehensive Time Series Data:** Create 180 days of sensor or network traffic data containing an upward trend, periodic seasonality, and injected spikes (anomalies).
5. **Compute Rolling Mean & Standard Deviation:** Calculate a 14-day rolling mean and rolling standard deviation to establish dynamic normal operating bounds.
6. **Detect Anomalies:** Flag data points as anomalies whenever the observed value exceeds $\text{Rolling Mean} \pm 2.5 \times \text{Rolling Std}$.
7. **Visualize Time Series & Bounds:** Plot the time series curve along with shaded confidence bands ($\pm 2.5\sigma$).
8. **Highlight Anomalies:** Overlay red scatter markers specifically on the detected anomalous timestamps.
9. **Export/Display Interactive Map:** Render the interactive Plotly map and save/show the rollover capabilities.
10. **Print Anomaly Summary:** Output the exact dates and values of the detected time series anomalies.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# --- PART 1: Map with Mouse Rollover Effect & User Interaction ---
map_data = pd.DataFrame({
    'City': ['Bengaluru', 'San Francisco', 'London', 'Tokyo', 'Berlin', 'Singapore', 'Sydney', 'Toronto'],
    'Country': ['India', 'USA', 'UK', 'Japan', 'Germany', 'Singapore', 'Australia', 'Canada'],
    'Latitude': [12.9716, 37.7749, 51.5074, 35.6762, 52.5200, 1.3521, -33.8688, 43.6532],
    'Longitude': [77.5946, -122.4194, -0.1278, 139.6503, 13.4050, 103.8198, 151.2093, -79.3832],
    'Active_Startups': [13500, 28000, 18500, 9200, 11000, 9800, 7400, 8900],
    'Funding_Billion_USD': [42.5, 115.0, 58.2, 24.1, 31.0, 35.8, 19.5, 22.4]
})

# Create interactive map with Mouse Rollover tooltips
fig_map = px.scatter_geo(
    map_data,
    lat='Latitude', lon='Longitude',
    size='Active_Startups',
    color='Funding_Billion_USD',
    hover_name='City',
    hover_data={'Country': True, 'Active_Startups': True, 'Funding_Billion_USD': True,
                'Latitude': False, 'Longitude': False},
    projection='natural earth',
    color_continuous_scale='Viridis',
    title='Interactive Map with Mouse Rollover: Global Tech Hubs (Zoom & Hover Enabled)'
)
fig_map.show()

# --- PART 2: Comprehensive Time Series Analysis (Trends, Patterns & Anomalies) ---
np.random.seed(10)
dates = pd.date_range(start='2025-01-01', periods=180, freq='D')
t = np.arange(180)
values = 200 + 0.8 * t + 30 * np.sin(2 * np.pi * t / 14) + np.random.normal(0, 8, 180)

# Inject artificial anomalies
values[45] = 360
values[110] = 140
values[155] = 410

ts = pd.DataFrame({'Metric': values}, index=dates)
ts['Rolling_Mean'] = ts['Metric'].rolling(window=14).mean()
ts['Rolling_Std'] = ts['Metric'].rolling(window=14).std()
ts['Upper_Bound'] = ts['Rolling_Mean'] + 2.5 * ts['Rolling_Std']
ts['Lower_Bound'] = ts['Rolling_Mean'] - 2.5 * ts['Rolling_Std']

# Identify Anomalies
ts['Anomaly'] = (ts['Metric'] > ts['Upper_Bound']) | (ts['Metric'] < ts['Lower_Bound'])
anomalies = ts[ts['Anomaly']]

# Plot Time Series with Anomalies
plt.figure(figsize=(13, 5))
plt.plot(ts.index, ts['Metric'], label='Observed Metric', color='steelblue')
plt.plot(ts.index, ts['Rolling_Mean'], label='14-Day Trend (Rolling Mean)', color='darkorange', linewidth=2)
plt.fill_between(ts.index, ts['Lower_Bound'], ts['Upper_Bound'], color='gray', alpha=0.2, label='Normal Band (±2.5 SD)')
plt.scatter(anomalies.index, anomalies['Metric'], color='red', s=90, zorder=5, label='Detected Anomaly')

plt.title('Comprehensive Time Series Analysis: Trends, Seasonality & Anomaly Detection')
plt.xlabel('Date')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("--- Detected Time Series Anomalies ---")
print(anomalies[['Metric', 'Rolling_Mean']])
```

---

## Question Set 14
**14.1** Apply Data analysis and representation on a Map using various Map datasets with Mouse Rollover effect.  
**14.2** Plan an Agriculture and Crop yield dataset and apply various EDA and visualization techniques and present an analysis report.

### Aim
To represent agricultural regional metrics on a map with mouse rollover effects, and to plan an Agriculture and Crop Yield dataset to perform Exploratory Data Analysis (EDA), visualize environmental factors affecting yield, and present an analytical report.

### Procedure
1. **Import Libraries:** Import `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, and `plotly.express`.
2. **Plan Agriculture & Crop Yield Dataset:** Construct a comprehensive dataset containing `Region`, `Crop_Type` (`Wheat`, `Rice`, `Maize`, `Cotton`), `Rainfall_mm`, `Fertilizer_kg_ha`, `Soil_Quality_Index`, `Yield_Tons_ha`, and geographic coordinates (`Latitude`, `Longitude`).
3. **Build Interactive Rollover Map:** Use `plotly.express.scatter_geo` scoped to Asia/India so hovering the mouse over each agricultural region displays `Region`, `Crop_Type`, `Rainfall_mm`, and `Yield_Tons_ha`.
4. **Perform Agricultural EDA:** Group the dataset by `Crop_Type` to evaluate mean yield, rainfall requirements, and fertilizer usage.
5. **Compute Feature Correlations:** Calculate the correlation between `Rainfall_mm`, `Fertilizer_kg_ha`, `Soil_Quality_Index`, and `Yield_Tons_ha`.
6. **Visualization 1 - Yield by Crop Type:** Create a boxplot or bar chart comparing `Yield_Tons_ha` across different crops.
7. **Visualization 2 - Fertilizer vs Yield Scatter:** Plot `Fertilizer_kg_ha` against `Yield_Tons_ha` colored by `Crop_Type`.
8. **Visualization 3 - Rainfall vs Yield Regression:** Plot the relationship between rainfall and yield.
9. **Render All Figures:** Display both the interactive rollover map and the Matplotlib/Seaborn statistical subplots.
10. **Present Crop Yield Report:** Print a formatted agricultural summary report with actionable agronomic insights.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- PART 1: Map Representation with Mouse Rollover Effect ---
agri_map_df = pd.DataFrame({
    'Region': ['Punjab', 'Tamil Nadu', 'Maharashtra', 'Uttar Pradesh', 'Gujarat', 'Andhra Pradesh', 'Madhya Pradesh'],
    'Primary_Crop': ['Wheat', 'Rice', 'Cotton', 'Wheat', 'Cotton', 'Rice', 'Maize'],
    'Latitude': [31.1471, 11.1271, 19.7515, 26.8467, 22.2587, 15.9129, 22.9734],
    'Longitude': [75.3412, 78.6569, 75.7139, 80.9462, 71.1924, 79.7400, 78.6569],
    'Avg_Yield_Tons_Ha': [4.8, 4.2, 2.9, 4.1, 3.2, 4.5, 3.6],
    'Annual_Rainfall_mm': [650, 920, 850, 800, 700, 950, 880]
})

fig_agri_map = px.scatter_geo(
    agri_map_df, lat='Latitude', lon='Longitude',
    color='Primary_Crop', size='Avg_Yield_Tons_Ha',
    hover_name='Region',
    hover_data=['Primary_Crop', 'Avg_Yield_Tons_Ha', 'Annual_Rainfall_mm'],
    scope='asia',
    title='Agricultural Map with Mouse Rollover: Crop Yield & Rainfall by Region'
)
fig_agri_map.update_geos(center=dict(lat=22, lon=79), projection_scale=4.5)
fig_agri_map.show()

# --- PART 2: Agriculture & Crop Yield Dataset EDA ---
np.random.seed(42)
n_farms = 120
crops = np.random.choice(['Rice', 'Wheat', 'Maize', 'Cotton'], n_farms)
rainfall = np.random.uniform(400, 1400, n_farms)
fertilizer = np.random.uniform(80, 250, n_farms)
soil_index = np.random.uniform(5.5, 9.0, n_farms)

# Simulate yield based on inputs
crop_yield = (0.002 * rainfall + 0.015 * fertilizer + 0.35 * soil_index + np.random.normal(0, 0.4, n_farms)).round(2)

df_crop = pd.DataFrame({
    'Farm_ID': range(1, n_farms + 1),
    'Crop_Type': crops,
    'Rainfall_mm': np.round(rainfall, 1),
    'Fertilizer_kg_ha': np.round(fertilizer, 1),
    'Soil_Quality_Index': np.round(soil_index, 2),
    'Yield_Tons_ha': crop_yield
})

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Plot 1: Boxplot of Yield by Crop Type
sns.boxplot(x='Crop_Type', y='Yield_Tons_ha', data=df_crop, palette='Set2', ax=axes[0])
axes[0].set_title('Crop Yield Distribution by Crop Type')
axes[0].set_ylabel('Yield (Tons / Hectare)')

# Plot 2: Scatter Plot (Fertilizer vs Yield)
sns.scatterplot(x='Fertilizer_kg_ha', y='Yield_Tons_ha', hue='Crop_Type', data=df_crop, s=60, ax=axes[1])
axes[1].set_title('Fertilizer Application vs Crop Yield')
axes[1].set_xlabel('Fertilizer (kg/ha)')
axes[1].set_ylabel('Yield (Tons/ha)')

# Plot 3: Regression Plot (Rainfall vs Yield)
sns.regplot(x='Rainfall_mm', y='Yield_Tons_ha', data=df_crop, color='forestgreen', ax=axes[2])
axes[2].set_title('Rainfall Impact on Crop Yield')
axes[2].set_xlabel('Rainfall (mm)')
axes[2].set_ylabel('Yield (Tons/ha)')

plt.tight_layout()
plt.show()

# Analysis Report
print("================ AGRICULTURE & CROP YIELD REPORT ================")
print(df_crop.groupby('Crop_Type')[['Rainfall_mm', 'Fertilizer_kg_ha', 'Soil_Quality_Index', 'Yield_Tons_ha']].mean().round(2))
print("\n--- Correlation with Yield (Tons/ha) ---")
print(df_crop[['Rainfall_mm', 'Fertilizer_kg_ha', 'Soil_Quality_Index', 'Yield_Tons_ha']].corr()['Yield_Tons_ha'].round(3))
print("=================================================================")
```

---

## Question Set 15
**15.1** Apply Data analysis and representation on a Map using various Map datasets with user interaction.  
**15.2** Utilize a Telecommunication and Network performance dataset and apply the various EDA and visualization techniques and present an analysis report.

### Aim
To build an interactive map supporting user-driven filtering, zooming, and selection on geographical datasets, and to analyze a Telecommunication and Network Performance dataset using Exploratory Data Analysis (EDA) and visualizations.

### Procedure
1. **Import Libraries:** Import `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, and `plotly.express`.
2. **Create Interactive Telecom Tower Map Dataset:** Build a geographic dataset of telecom hubs across cities with `City`, `Network_Type` (`5G`, `4G LTE`), `Signal_Strength_dBm`, `Active_Users`, and `Latency_ms`.
3. **Configure Map User Interaction:** Use `plotly.express.scatter_mapbox` (with open-street-map style) to allow users to interactively pan, zoom, toggle network layers via the legend, and inspect nodes.
4. **Construct Telecom & Network Performance Dataset:** Generate 150 network cell records containing `Tower_ID`, `Band` (`5G`, `4G`, `Fiber`), `Download_Mbps`, `Upload_Mbps`, `Latency_ms`, `Packet_Loss_Pct`, and `Call_Drop_Rate_Pct`.
5. **Perform Telecom EDA:** Group records by `Band` to compare throughput speeds, latency, and packet loss.
6. **Visualization 1 - Throughput Comparison:** Plot a bar or violin plot comparing `Download_Mbps` across `5G`, `4G`, and `Fiber`.
7. **Visualization 2 - Latency vs Download Speed:** Create a scatter plot illustrating the inverse relationship between `Latency_ms` and `Download_Mbps`.
8. **Visualization 3 - Packet Loss vs Call Drop Rate:** Plot a joint/regression chart examining how packet loss drives call drops.
9. **Render Visuals:** Display the interactive map and the 3-panel network performance figure.
10. **Generate Telecom Analysis Report:** Print a network quality-of-service (QoS) summary report.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- PART 1: Map Representation with User Interaction ---
tower_map_df = pd.DataFrame({
    'City': ['Chennai', 'Coimbatore', 'Madurai', 'Tiruchirappalli', 'Salem', 'Bengaluru', 'Hyderabad', 'Kochi'],
    'Network_Band': ['5G', '5G', '4G LTE', '4G LTE', '4G LTE', '5G', '5G', '5G'],
    'Latitude': [13.0827, 11.0168, 9.9252, 10.7905, 11.6643, 12.9716, 17.3850, 9.9312],
    'Longitude': [80.2707, 76.9558, 78.1198, 78.7047, 78.1460, 77.5946, 78.4867, 76.2673],
    'Avg_Download_Mbps': [320, 290, 65, 58, 52, 350, 310, 275],
    'Active_Subscribers_K': [850, 520, 410, 330, 290, 980, 890, 460]
})

# Interactive OpenStreetMap with zoom, pan, and legend filtering
fig_interactive_map = px.scatter_mapbox(
    tower_map_df, lat='Latitude', lon='Longitude',
    color='Network_Band', size='Active_Subscribers_K',
    hover_name='City',
    hover_data=['Avg_Download_Mbps', 'Active_Subscribers_K'],
    zoom=5, height=450,
    mapbox_style='open-street-map',
    title='Interactive Telecom Network Map (Pan, Zoom & Click Legend to Filter)'
)
fig_interactive_map.show()

# --- PART 2: Telecommunication & Network Performance EDA ---
np.random.seed(42)
n_nodes = 150
bands = np.random.choice(['4G LTE', '5G', 'Fiber'], n_nodes, p=[0.4, 0.4, 0.2])

download_speed = np.where(bands == '5G', np.random.normal(300, 45, n_nodes),
                 np.where(bands == 'Fiber', np.random.normal(450, 60, n_nodes),
                          np.random.normal(55, 15, n_nodes)))

latency = np.where(bands == '5G', np.random.normal(15, 4, n_nodes),
          np.where(bands == 'Fiber', np.random.normal(8, 2, n_nodes),
                   np.random.normal(42, 10, n_nodes)))

packet_loss = np.clip(np.random.exponential(0.8, n_nodes), 0.05, 5.0)
call_drop_rate = np.clip(0.3 + 0.65 * packet_loss + np.random.normal(0, 0.2, n_nodes), 0.1, 6.0)

df_telecom = pd.DataFrame({
    'Node_ID': range(1001, 1001 + n_nodes),
    'Network_Type': bands,
    'Download_Mbps': np.round(download_speed, 1),
    'Latency_ms': np.round(latency, 1),
    'Packet_Loss_Pct': np.round(packet_loss, 2),
    'Call_Drop_Rate_Pct': np.round(call_drop_rate, 2)
})

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# Plot 1: Download Speed by Network Type
sns.boxplot(x='Network_Type', y='Download_Mbps', data=df_telecom, palette='cool', ax=axes[0])
axes[0].set_title('Download Speed by Network Technology')
axes[0].set_ylabel('Download Speed (Mbps)')

# Plot 2: Latency vs Download Speed
sns.scatterplot(x='Latency_ms', y='Download_Mbps', hue='Network_Type', data=df_telecom, s=60, ax=axes[1])
axes[1].set_title('Network Latency vs Download Throughput')
axes[1].set_xlabel('Latency (ms)')
axes[1].set_ylabel('Download Speed (Mbps)')

# Plot 3: Packet Loss vs Call Drop Rate
sns.regplot(x='Packet_Loss_Pct', y='Call_Drop_Rate_Pct', data=df_telecom, color='crimson', ax=axes[2])
axes[2].set_title('Impact of Packet Loss on Call Drop Rate')
axes[2].set_xlabel('Packet Loss (%)')
axes[2].set_ylabel('Call Drop Rate (%)')

plt.tight_layout()
plt.show()

# Analysis Report
print("======== TELECOMMUNICATION & NETWORK PERFORMANCE REPORT ========")
print(df_telecom.groupby('Network_Type')[['Download_Mbps', 'Latency_ms', 'Packet_Loss_Pct', 'Call_Drop_Rate_Pct']].mean().round(2))
print("================================================================")
```

---

## Question Set 16
**16.1** Choose a Water resource management dataset and apply various EDA and visualization techniques and present an analysis report.  
**16.2** Plan a Waste management and Recycling dataset and apply various EDA and visualization techniques and present an analysis report.

### Aim
To construct and analyze both a Water Resource Management dataset and a Waste Management & Recycling dataset using Exploratory Data Analysis (EDA), comparative visualizations, and sustainability analysis reports.

### Procedure
1. **Import Libraries:** Import `numpy`, `pandas`, `matplotlib.pyplot`, and `seaborn`.
2. **Create Water Resource Management Dataset:** Construct a DataFrame representing reservoirs/basins with attributes `Reservoir_Name`, `Capacity_BCM`, `Current_Storage_BCM`, `Water_Quality_pH`, `Dissolved_Oxygen_mgL`, and `Consumption_Sector` (`Agricultural`, `Industrial`, `Domestic`).
3. **Compute Water Storage Percentage:** Derive `Storage_Fill_Pct` as $(\text{Current Storage} / \text{Capacity}) \times 100$.
4. **Visualize Water Resource Data:** Plot reservoir storage levels as a horizontal bar chart and water quality (`pH` vs `Dissolved_Oxygen`) as a scatter plot.
5. **Plan Waste Management & Recycling Dataset:** Build a municipal waste dataset across zones containing `City_Zone`, `Total_Waste_Tons`, `Recycled_Tons`, `Composted_Tons`, `Landfill_Tons`, and `Waste_Category` (`Organic`, `Plastic`, `E-Waste`, `Metal`).
6. **Compute Recycling Efficiency Rate:** Calculate `Recycling_Rate_Pct` as $((\text{Recycled} + \text{Composted}) / \text{Total Waste}) \times 100$.
7. **Visualize Waste Management Data:** Construct a stacked bar chart comparing `Recycled_Tons`, `Composted_Tons`, and `Landfill_Tons` across zones, plus a pie chart of waste composition.
8. **Statistical Aggregation:** Calculate summary statistics for reservoir storage and municipal recycling rates.
9. **Render Multi-Panel Dashboard:** Display the 2x2 visualization grid covering both environmental domains.
10. **Present Combined Sustainability Report:** Output structured reports for both Water Resource Management and Waste Management & Recycling.

### Program
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- PART 1: Water Resource Management Dataset ---
water_data = {
    'Reservoir': ['Mettur', 'Bhavanisagar', 'Vaigai', 'Papanasam', 'Aliyar', 'Chembarambakkam', 'Krishnagiri'],
    'Capacity_TMC': [93.4, 32.8, 6.1, 5.5, 3.8, 3.6, 1.6],
    'Current_Storage_TMC': [78.2, 26.5, 4.2, 4.8, 3.1, 2.9, 1.1],
    'Water_pH': [7.4, 7.6, 7.1, 6.9, 7.3, 7.8, 7.5],
    'Dissolved_Oxygen_mgL': [6.8, 7.2, 5.9, 7.5, 6.9, 5.4, 6.2]
}
df_water = pd.DataFrame(water_data)
df_water['Fill_Percentage'] = ((df_water['Current_Storage_TMC'] / df_water['Capacity_TMC']) * 100).round(1)

# --- PART 2: Waste Management and Recycling Dataset ---
waste_data = {
    'Zone': ['North Zone', 'South Zone', 'East Zone', 'West Zone', 'Central Zone'],
    'Total_Waste_Tons': [450, 520, 390, 480, 610],
    'Recycled_Tons': [140, 195, 110, 175, 230],
    'Composted_Tons': [110, 140, 95, 125, 160],
    'Landfill_Tons': [200, 185, 185, 180, 220]
}
df_waste = pd.DataFrame(waste_data)
df_waste['Recovery_Rate_Pct'] = (((df_waste['Recycled_Tons'] + df_waste['Composted_Tons']) / df_waste['Total_Waste_Tons']) * 100).round(1)

# --- Visualizations (2x2 Grid) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Water Reservoir Fill Percentage
axes[0, 0].barh(df_water['Reservoir'], df_water['Fill_Percentage'], color='dodgerblue', edgecolor='black')
axes[0, 0].set_title('Water Resource Management: Reservoir Fill Level (%)')
axes[0, 0].set_xlabel('Storage Fill Percentage (%)')
axes[0, 0].set_xlim(0, 100)

# Plot 2: Water Quality (pH vs Dissolved Oxygen)
axes[0, 1].scatter(df_water['Water_pH'], df_water['Dissolved_Oxygen_mgL'], color='teal', s=100, edgecolors='black')
for _, row in df_water.iterrows():
    axes[0, 1].annotate(row['Reservoir'], (row['Water_pH'] + 0.02, row['Dissolved_Oxygen_mgL'] + 0.03), fontsize=9)
axes[0, 1].set_title('Water Quality Analysis: pH vs Dissolved Oxygen')
axes[0, 1].set_xlabel('Water pH Level')
axes[0, 1].set_ylabel('Dissolved Oxygen (mg/L)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Stacked Bar Chart of Waste Management by Zone
axes[1, 0].bar(df_waste['Zone'], df_waste['Recycled_Tons'], label='Recycled', color='#2ecc71')
axes[1, 0].bar(df_waste['Zone'], df_waste['Composted_Tons'], bottom=df_waste['Recycled_Tons'], label='Composted', color='#f1c40f')
axes[1, 0].bar(df_waste['Zone'], df_waste['Landfill_Tons'],
               bottom=df_waste['Recycled_Tons'] + df_waste['Composted_Tons'], label='Landfill', color='#e74c3c')
axes[1, 0].set_title('Waste Management Disposition Across City Zones')
axes[1, 0].set_ylabel('Waste Volume (Tons)')
axes[1, 0].legend()

# Plot 4: Recovery Rate Percentage by Zone
sns.barplot(x='Zone', y='Recovery_Rate_Pct', data=df_waste, palette='Greens_d', ax=axes[1, 1])
axes[1, 1].set_title('Recycling & Composting Recovery Rate (%) by Zone')
axes[1, 1].set_ylabel('Recovery Rate (%)')
axes[1, 1].set_ylim(0, 100)

plt.tight_layout()
plt.show()

# Step 10: Analysis Reports
print("================ WATER RESOURCE MANAGEMENT REPORT ================")
print(df_water[['Reservoir', 'Capacity_TMC', 'Current_Storage_TMC', 'Fill_Percentage', 'Water_pH', 'Dissolved_Oxygen_mgL']])
print(f"\nAverage Reservoir Fill Level: {df_water['Fill_Percentage'].mean():.1f}%")
print(f"Mean Water pH: {df_water['Water_pH'].mean():.2f} | Mean Dissolved Oxygen: {df_water['Dissolved_Oxygen_mgL'].mean():.2f} mg/L")

print("\n================ WASTE MANAGEMENT & RECYCLING REPORT ================")
print(df_waste[['Zone', 'Total_Waste_Tons', 'Recycled_Tons', 'Composted_Tons', 'Landfill_Tons', 'Recovery_Rate_Pct']])
print(f"\nTotal Municipal Waste Generated: {df_waste['Total_Waste_Tons'].sum()} Tons")
print(f"Overall City Waste Recovery Rate: {df_waste['Recovery_Rate_Pct'].mean():.1f}%")
print("=====================================================================")
```