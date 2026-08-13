import pandas as pd
from typing import Dict

def task_1_load_and_inspect(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from the specified CSV file.
    """
    try:
        df = pd.read_csv(file_path) 
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise

def task_2_check_dataset_health(df: pd.DataFrame) -> Dict[str, int]:
    """
    Checks total rows and missing values.
    """
    total_tweets = len(df)
    missing_tweets = df['tweet'].isnull().sum()
    missing_classes = df['class'].isnull().sum()
    return {
        "total_tweets": int(total_tweets),
        "missing_tweets": int(missing_tweets),
        "missing_classes": int(missing_classes)
    }

def task_3_analyze_class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Counts the tweets of each class.
    """
    return df['class'].value_counts() 

def task_4_analyze_tweet_length(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes average word count for each class and overall.                            
    """
    # Count words in each tweet (split on whitespace)
    df['word_count'] = df['tweet'].apply(lambda x: len(str(x).split()))
    overall_avg = df['word_count'].mean()
    avg_hate = df[df['class'] == 0]['word_count'].mean()
    avg_off = df[df['class'] == 1]['word_count'].mean()
    avg_neither = df[df['class'] == 2]['word_count'].mean()
    return {
        "overall_avg_word_count": overall_avg,
        "avg_hate_word_count": avg_hate,
        "avg_offensive_word_count": avg_off,
        "avg_neither_word_count": avg_neither
    }



if __name__ == "__main__":
    try:
        FILE_PATH = 'hate_speech.csv'
        
        # Task 1
        df = task_1_load_and_inspect(FILE_PATH)
        print("Task 1: Data Loaded")
        print(df.head())
        print("\n")
        
        # Task 2
        print("Task 2: Checking Dataset Health")
        health = task_2_check_dataset_health(df)
        for key, val in health.items():
            print(f"{key.replace('_', ' ').title()}: {val}")
        print("\n")
        
        # Task 3
        print("Task 3: Class Distribution")
        dist = task_3_analyze_class_distribution(df)
        print("Tweet counts per class:")
        print(dist)
        is_balanced = "Yes" if abs(dist.get(0,0)-dist.get(1,0)) < 0.05*len(df) else "No"
        print(f"\nIs the dataset balanced (roughly)? {is_balanced}")
        print("\n")
        
        # Task 4
        print("Task 4: Tweet Length Analysis")
        length_stats = task_4_analyze_tweet_length(df)
        print(f"Overall Average Word Count: {length_stats['overall_avg_word_count']:.2f} words")
        print(f"Average Hate-speech Word Count: {length_stats['avg_hate_word_count']:.2f} words")
        print(f"Average Offensive Word Count: {length_stats['avg_offensive_word_count']:.2f} words")
        print(f"Average Neutral Word Count: {length_stats['avg_neither_word_count']:.2f} words")
        
    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
