# Machine learning model that can accurately predict if a borrower will pay off their loan on time or not?

`PART 1: Data cleaning`
--------------------------------------------
[Lending Club](https://www.lendingclub.com/info/download-data.action) releases data for all of the approved and declined loan applications periodically on their website. We have dataset for years from 2007 to 2011. **Data dictionary** can be found on this [google drive](https://docs.google.com/spreadsheets/d/191B2yJ4H1ZPXq0_ByhUgWMFZOYem5jFz0Y3by_7YBY4/edit). The dataset is attached in this repository.

PROBLEM STATEMENT : `Can we build a machine learning model that can accurately predict if a borrower will pay off their loan on time or not?`
--------------------------------------------------------------
## So Far...
* We initially had dataset of shape (42538,52) and then we analysed and cleaned it to bring it to the shape (38770,23)
* We removed columns that may leak information or the columns that aren't useful for our modelling purpose
* We decided our target column and decided to focus on modelling efforts based on Binary Classification

`PART 2: Features Preparation`
--------------------------------------------
Here we will be continuting our prediction modelling. We will use the csv file we saved in *PART 1*.

In this part we will mainly focus on preparing features. We will prepare data for machine learning by focusing on handling missing values, converting categorical values to numeric values and removing any extraneous columns we encounter. We need to convert categorical type columns to numerical type because most of the Machine Learning algorithms assume data is numeric and contains no missing values.If this requirement isn't fulfilled then sklearn will raise error when working with models like `LinearRegression` and `LogisticRegression`.

## So far..
* We have converted necessary columns to numerical type.
* Removed columns which provide overlapping information
* Added new features using dummy variables
* Cleaned dataset by removing null values
* Mapped category values to specific integer

We have done a lot of preprocessing till now. Dataset looks good and cleaned. We will now start working on Machine Learning models in next part.

`PART 3: Applying ML Algorithm`
--------------------------------------------

In last two notebooks we cleaned data. Our eventual goal is to generate features from data, which we can feed into Machine Learning algorithm. The algorithm will make predictions whether or not loan will be paid off in time or not, which is contained in `loan_status` column of dataset. We prepared data, we cleaned the data, we removed columns containing data that can result into leakage, columns which have zero variance and columns which had redundant information. We also cleaned columns with formatting issues and converted categorical columns to dummy variable.

#### class imbalance:
We know there is class imbalance as number of `1` in **loan_status** column is 6x more than number of `0`. We need to be aware of that as it may impact prediction. Machine learning models may show high accuracy in such case for training set but they aren't actually learning anything from train set.

## Conclusion:
Using random forest classifier ddn't improve our false positive rate. The model is likely weighting too heavily on `1` class and still predicting `1s`.We can further apply penalties.

Our best model so far had true positive rate of `25%` and false positive rate of `9%`.

We can futher tune our models for better predictions.
