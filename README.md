# Twitter/X Data Mining with Python

A Python-based project for collecting, preprocessing, and analysing social-media data from Twitter/X. The project demonstrates a complete text-mining workflow, starting from raw posts and progressing through text preprocessing, frequency analysis, term co-occurrence, geolocation analysis, and sentiment classification.

## Overview

Social-media platforms generate large volumes of unstructured text. This project explores how Python can be used to transform raw Twitter/X posts into structured information that can be analysed for patterns and insights.

The workflow covers:

* Data collection from Twitter/X
* Text cleaning and tokenisation
* Unigram and bigram frequency analysis
* Target-term co-occurrence analysis
* User-location analysis
* Sentiment classification using VADER
* Basic evaluation against existing sentiment labels

The project was demonstrated using a dataset of Twitter/X posts related to **rugby and the World Cup**.

## Pipeline

```text
Twitter/X Data
      │
      ▼
Data Collection
      │
      ▼
Raw Tweet Dataset
      │
      ▼
Text Pre-processing
      │
      ├── URL Removal
      ├── Mention Removal
      ├── Hashtag Processing
      ├── Lowercasing
      ├── Tokenisation
      └── Stop-word Filtering
      │
      ▼
Text Analysis
      │
      ├── Unigram Frequency
      ├── Bigram Frequency
      └── Term Co-occurrence
      │
      ├───────────────┐
      ▼               ▼
Geolocation       Sentiment Analysis
Analysis          using VADER
      │               │
      └───────┬───────┘
              ▼
        Extracted Insights
```

## Features

### 1. Data Collection

The project demonstrates three approaches for obtaining Twitter/X data:

**Alternative scraping method**

A scraping script is maintained as an alternative data-collection approach when API access is unavailable.

**Apify**

Apify is used as a hosted scraping platform for larger historical data collection. The documented collection used:

* Search query: `world cup`
* Retweets/replies excluded
* Maximum: 100 tweets
* Sorting: Most Recent
* Date range: June 1, 2026 – August 29, 2026

**X API v2**

The project also demonstrates direct collection through the official X API v2 using a Bearer Token. The resulting JSON contains tweet text, metadata, and engagement information.

## 2. Text Pre-processing

Raw social-media text contains URLs, mentions, hashtags, punctuation, and other elements that can interfere with text analysis.

The preprocessing pipeline:

1. Removes URLs
2. Removes `@mentions`
3. Removes the `#` symbol
4. Removes `RT`
5. Converts text to lowercase
6. Removes non-alphabetic characters
7. Tokenises the text
8. Removes English stop words
9. Removes very short tokens

Example:

```text
Raw:
That performance from Ireland was class from start to finish,
14 points and never looked back #RugbyUnion #SixNations

Tokens:
[
    'performance',
    'ireland',
    'class',
    'start',
    'finish',
    'points',
    'never',
    'looked',
    'back',
    'rugbyunion',
    'sixnations'
]
```

The cleaned tokens are stored in a new dataframe column:

```python
df['clean_tokens'] = df['text'].apply(clean_tweet)
```

## 3. Term Frequency Analysis

The project uses `Counter` and NLTK's `ngrams()` functionality to analyse frequently occurring terms.

### Unigrams

A unigram represents a single word.

```python
from collections import Counter

all_tokens = [
    token
    for sublist in df['clean_tokens']
    for token in sublist
]

unigram_freq = Counter(all_tokens)
```

The documented top five unigrams were:

| Term       | Frequency |
| ---------- | --------: |
| ireland    |      1452 |
| england    |      1321 |
| wales      |       983 |
| rugby      |       912 |
| sixnations |       870 |

### Bigrams

A bigram represents two consecutive words.

```python
from nltk.util import ngrams

all_bigrams = []

for tokens in df['clean_tokens']:
    all_bigrams.extend(list(ngrams(tokens, 2)))

bigram_freq = Counter(all_bigrams)
```

Documented top five bigrams:

| Bigram        | Frequency |
| ------------- | --------: |
| six nations   |       810 |
| full time     |       420 |
| final try     |       312 |
| coach delaney |       280 |
| brilliant win |       245 |

These frequencies provide a simple way to identify dominant terms and recurring phrases within the dataset.

## 4. Target-Term Co-occurrence

Co-occurrence analysis identifies words that frequently appear in the same tweets as a selected target term.

In the example, the target term is:

```python
target = "ireland"
```

Tweets containing the target are selected, and the other words occurring within those tweets are counted.

```python
target_tweets = df[
    df['clean_tokens'].apply(lambda x: target in x)
]

co_occurrence = Counter()

for tokens in target_tweets['clean_tokens']:
    for word in tokens:
        if word != target:
            co_occurrence[word] += 1
```

The documented co-occurring terms were:

| Term     | Co-occurrence |
| -------- | ------------: |
| england  |           432 |
| match    |           265 |
| win      |           240 |
| scotland |           215 |
| defence  |           175 |

This provides a simple way to examine which concepts are associated with a particular target term within the dataset.

## 5. Geolocation Analysis

The dataset contains a `user_location` field that can be used to identify the most frequently represented user locations.

```python
loc = df['user_location'].value_counts()

top = loc[loc >= 10].head(10)

for city, count in top.items():
    print(f"{city}: {count} tweets")
```

Documented results:

| Location   | Tweets |
| ---------- | -----: |
| Scotland   |    320 |
| Dublin     |    280 |
| Paris      |    250 |
| France     |    210 |
| London     |    190 |
| Manchester |    160 |
| Nice       |    140 |
| Cork       |    120 |
| Limerick   |    110 |
| Swansea    |    100 |

The analysis filters locations to those appearing in at least 10 tweets.

## 6. Sentiment Analysis

The project uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)** to classify tweets into three categories:

* Positive
* Neutral
* Negative

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_label(text):
    score = analyzer.polarity_scores(text)['compound']

    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
```

The sentiment label is then generated for every tweet:

```python
df['vader'] = df['text'].apply(get_label)
```

### Documented distribution

| Sentiment | Tweets |
| --------- | -----: |
| Positive  |   1090 |
| Neutral   |    470 |
| Negative  |    440 |

The article also reports that VADER agreed with the dataset's existing sentiment labels for **77.0%** of the records.

## Technologies Used

* **Python**
* **Pandas** — dataset manipulation
* **NLTK** — tokenisation, stop words and n-grams
* **VADER Sentiment** — sentiment analysis
* **Matplotlib** — visualisation
* **Seaborn** — visualisation
* **WordCloud** — text visualisation
* **Regular Expressions (`re`)** — text cleaning
* **Apify** — hosted data collection
* **X API v2** — direct Twitter/X data collection

## Installation

Clone the repository:

```bash
git clone https://github.com/saif-pixel1/twitter-data-mining-with-python.git
cd twitter-data-mining-with-python
```

Install the Python dependencies:

```bash
pip install pandas matplotlib seaborn nltk wordcloud vaderSentiment
```

Download the required NLTK resources:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

## Dataset

The project works with Twitter/X posts containing fields such as tweet text, user location, and sentiment labels.

Depending on the collection method, data can be obtained as:

```text
CSV
JSON
```

For API-based collection, the project saves the raw response as:

```text
dataset.json
```

The repository also contains a dataset file used by the alternative collection workflow.

## Project Structure

A suggested repository structure is:

```text
twitter-data-mining-with-python/
│
├── data/
│   ├── data.csv
│   └── dataset.json
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── collect_data.py
│   ├── preprocessing.py
│   ├── frequency_analysis.py
│   ├── cooccurrence.py
│   ├── geolocation.py
│   └── sentiment.py
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Example Workflow

```python
# Load dataset
import pandas as pd

df = pd.read_csv("data/data.csv")

# Clean tweets
df['clean_tokens'] = df['text'].apply(clean_tweet)

# Frequency analysis
unigram_freq = Counter(
    token
    for tokens in df['clean_tokens']
    for token in tokens
)

print(unigram_freq.most_common(10))
```

The same cleaned dataset can then be passed to the n-gram, co-occurrence, geolocation, and sentiment-analysis stages.

## Results

The project demonstrates that a relatively small Python pipeline can transform raw social-media posts into several forms of structured analysis:

* Most frequent words
* Most frequent word combinations
* Words associated with a selected topic
* Distribution of user locations
* Positive, neutral, and negative sentiment categories
* Agreement between VADER and existing sentiment labels

The documented sentiment experiment reported a 77.0% agreement between VADER predictions and the original labels.

## Limitations

This project is intended as a practical demonstration of Twitter/X text mining rather than a comprehensive NLP system.

Important limitations include:

* The documented dataset is limited in size.
* Search-based collection does not represent all Twitter/X activity.
* User-provided location fields may not correspond to precise geographic locations.
* VADER is a rule-based sentiment method and may struggle with sarcasm, context, or domain-specific language.
* Co-occurrence indicates that terms appear together; it does not establish a causal or semantic relationship.
* API and scraping availability can change over time.

## Future Improvements

Possible extensions include:

* Larger-scale data collection
* Automated data pipelines
* Hashtag trend detection
* Topic modelling
* Named Entity Recognition
* Transformer-based sentiment classification
* Network analysis of users and interactions
* Interactive dashboards
* Real-time Twitter/X stream processing
* More robust geographic analysis

## Author

**Saif Akhtar**

AI / ML Researcher

GitHub: `@saif-pixel1`

Email: `saifakhtar43528@gmail.com`

## Project Status

**Educational / Research Project**

This repository demonstrates an end-to-end workflow for Twitter/X data mining using Python, from data collection through preprocessing and exploratory analysis to basic sentiment classification.
