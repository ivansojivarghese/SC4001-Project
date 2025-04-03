# SC4001-Project

## Exploratory Data Analysis (EDA)

The process of Exploratory Data Analysis (EDA) in this code focuses on understanding the data, its structure, and any potential patterns or problems that may need to be addressed before model training or further analysis. The EDA process is crucial in machine learning and data science because it allows you to uncover insights from the dataset that might affect your model's performance.

Here's a detailed breakdown of the steps in the EDA process for the IMDB and SST-2 datasets, explaining why the data was cleaned and handled in these specific ways:

### Step 1: Data Loading and Preprocessing

#### Load Dataset:

The datasets are loaded into memory using the pandas library (imdb_df and sst2_df). Both datasets come in CSV or dataset format, and they contain reviews and labels (sentiment labels for IMDB and SST-2).

#### Why Clean the Data:

These datasets often contain extra columns that are not required for analysis (like idx in SST-2).

We also need to preprocess the text (tokenization, lemmatization) to ensure the text data is usable for model training and analysis.

##### Preprocessing Text Data:

Lowercasing: Converts all the text to lowercase to standardize the data. This ensures that "Good" and "good" are considered the same word.

Removing Punctuation: This is done to clean the text of any non-informative symbols.

Lemmatization: Words are reduced to their base form (e.g., "running" becomes "run") to standardize the vocabulary and ensure that variants of the same word are counted as one.

Removing Stop Words: Words like "the", "a", "and", etc., are removed since they do not contribute to the sentiment analysis (these are words that are very frequent but carry little meaning).

#### Why this Cleaning:

Text cleaning helps to standardize the data and reduces noise, making it easier to analyze patterns related to sentiment or other features.

It also reduces the size of the dataset by removing unnecessary words (stopwords) and punctuation, improving the model's focus on relevant data.

### Step 2: Tokenization and Frequency Distribution

After preprocessing, the text is tokenized and analyzed for the most frequent words in positive and negative reviews.

#### Tokenization:

This process splits the review text into individual words (tokens), which are the smallest units of meaningful text.

#### Frequency Count:

Using the Counter class from the collections module, we calculate the frequency of words in positive and negative reviews for both IMDB and SST-2 datasets.

This step helps in identifying the most common words in the reviews, which might be indicative of the review sentiment.

##### Why this Cleaning:

By separating tokens (words) and looking at the most common words in positive and negative reviews, we gain valuable insights into what terms are most associated with each sentiment class.

This helps identify patterns (e.g., the presence of certain words like "good" in positive reviews or "boring" in negative reviews) and could assist in feature selection or feature engineering for model training.

### Step 3: Feature Engineering
#### Review Length:

A new feature, review_length, is created to store the length of each review (i.e., the number of characters in the review).

This feature is useful because the length of a review can sometimes correlate with sentiment. For instance, longer reviews might be more detailed, and certain sentiments may correlate with review length.

#### Why this Cleaning:

Creating the review_length feature allows for deeper analysis of the reviews. For example, very short reviews might be more likely to be polarizing, while longer reviews might show more balanced sentiment.

By incorporating this feature, we could later use it for visualization or as an additional input feature in a machine learning model.

### Step 4: Class Distribution
Class distribution refers to the balance between the different classes in the dataset, i.e., positive vs. negative sentiment.

#### Count Plot:

A count plot is generated to visualize the distribution of sentiment labels. This is done using sns.countplot(), which shows how many positive and negative reviews are present in the dataset.

#### Why this Cleaning:

Understanding the class distribution is crucial to identify any potential class imbalance (i.e., one class being more frequent than the other). An imbalance could affect model performance, as the model might become biased toward the majority class.

This allows the data scientist to make decisions about whether resampling techniques (like oversampling or undersampling) might be needed.

### Step 5: Sentence or Review Length Distribution
#### Histogram:

A histogram is plotted to show the distribution of review lengths (review_length feature). This helps us understand how long the reviews are on average and if there is any skewness in the lengths of the reviews.

#### Why this Cleaning:

The distribution of review lengths provides insights into the nature of the reviews. For example, if the reviews tend to be very short or very long, it might suggest different patterns in sentiment or user behavior.

This information might also help determine thresholds for truncating or padding reviews during preprocessing for machine learning models.

### Step 6: Saving Cleaned Data
After cleaning and performing EDA, the cleaned data is saved into new CSV files (IMDB_Dataset_Cleaned.csv and SST2_Dataset_Cleaned.csv).

### Summary of Why the Data Was Cleaned This Way:
Preprocessing: Text cleaning (lowercasing, punctuation removal, lemmatization, and stopwords removal) standardizes the text and makes it more consistent, allowing for better analysis and model performance.

Tokenization: Breaking the text into words helps in understanding the frequency and significance of individual words.

Class Distribution & Length Distribution: Helps identify any biases or trends in the dataset (e.g., imbalances or lengths of reviews that may require further attention).

Column Renaming and Dropping: Improves readability and consistency, preparing the data for use in machine learning models and analyses.

Saving Cleaned Data: Ensures that the data is saved in a clean and structured format for easy use in future tasks.

This process ensures that the dataset is prepared in an efficient and structured manner, which will lead to better insights, visualizations, and ultimately improved machine learning models.

