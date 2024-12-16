import os
import time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from prettytable import PrettyTable
from deep_translator import GoogleTranslator
import speech_recognition as sr
import re
from langdetect import detect
from collections import Counter
from termcolor import colored
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def translate_text(text, target_language="en"):
    """
    Translates input text to the target language (default: English).
    """
    try:
        detected_language = detect(text)
        if detected_language == target_language:
            return text  # No translation needed
        translated = GoogleTranslator(source="auto", target=target_language).translate(text)
        return translated
    except Exception as e:
        print(colored(f"Translation failed: {e}", 'red'))
        return text


def capture_voice_input(language="en-IN"):  # Default is set to English (India)
    """
    Captures voice input and converts it to text using Google Speech Recognition.
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print(colored(f"🎙️ Listening for input in {language}...", 'yellow'))
            audio = recognizer.listen(source)
            print(colored("🔄 Processing audio...", 'yellow'))
            text = recognizer.recognize_google(audio, language=language)
            return text
    except sr.UnknownValueError:
        print(colored("❌ Could not understand the audio.", 'red'))
    except sr.RequestError as e:
        print(colored(f"❌ Could not request results: {e}", 'red'))
    return None


def analyze_textblob(text):
    """
    Performs sentiment analysis using TextBlob.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"
    return sentiment, polarity, subjectivity


def analyze_vader(text):
    """
    Performs sentiment analysis using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    sentiment = "Positive" if compound > 0.2 else "Negative" if compound < -0.2 else "Neutral"
    return sentiment, compound


def analyze_huggingface(text):
    """
    Performs sentiment analysis using HuggingFace transformers.
    """
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = sentiment_analyzer(text)[0]
        sentiment = result["label"].capitalize()
        confidence = result["score"]
        return sentiment, confidence
    except Exception as e:
        print(colored(f"🤖 HuggingFace analysis failed: {e}", 'red'))
        return "Error", None


def extract_keywords(text):
    """
    Extracts the most common keywords from the input text.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    return [keyword for keyword, _ in word_counts.most_common(10)]


def plot_sentiment_scores(textblob_score, vader_score, huggingface_score):
    """
    Plots horizontal bar chart for sentiment scores.
    """
    models = ["TextBlob", "VADER", "HuggingFace"]
    scores = [textblob_score, vader_score, huggingface_score]
    colors = ['green' if score > 0 else 'red' for score in scores]

    plt.figure(figsize=(8, 6))
    plt.barh(models, scores, color=colors)
    plt.xlabel('Sentiment Score')
    plt.title('Sentiment Analysis Scores by Model')
    plt.tight_layout()
    plt.show()


def display_wordcloud(keywords, sentiment="neutral"):
    """
    Generates and displays a word cloud for the extracted keywords.
    """
    text = " ".join(keywords)
    colormap = 'coolwarm' if sentiment.lower() == "positive" else 'coolwarm_r' if sentiment.lower() == "negative" else 'Blues'

    wordcloud = WordCloud(
        width=800, height=400, background_color='white', colormap=colormap,
        max_words=100, contour_color='steelblue', contour_width=1.5
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Keyword Word Cloud - {sentiment.capitalize()} Sentiment", fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_sentiment_scores(textblob_score, vader_score, huggingface_score):
    """
    Plots pie chart for sentiment scores, ensuring no negative values.
    """
    # Normalize sentiment scores (make them non-negative)
    scores = [max(score, 0) for score in [textblob_score, vader_score, huggingface_score]]

    # Handle the case where all scores are zero (no sentiment)
    if all(score == 0 for score in scores):
        print(colored("❌ All sentiment scores are zero, cannot plot pie chart.", 'red'))
        return

    models = ["TextBlob", "VADER", "HuggingFace"]
    
    # Define colors based on sentiment
    colors = ['#66b3ff', '#ff9999', '#99ff99']  # Light Blue, Light Red, Light Green

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(scores, labels=models, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title("Sentiment Distribution", fontsize=16)
    plt.show()

def print_slow(text, speed=0.05):
    """
    Prints text with a typing animation.
    """
    for char in text:
        print(char, end='', flush=True)
        time.sleep(speed)
    print()


def display_results(text, textblob_result, vader_result, huggingface_result, keywords):
    """
    Displays sentiment analysis results in a PrettyTable with emojis.
    """
    print_slow(colored("\n--- Sentiment Analysis Results --- 📝", 'cyan', attrs=['bold']))

    table = PrettyTable()
    table.field_names = ["Model", "Sentiment", "Score/Polarity", "Additional Info"]

    sentiment_emoji = {
        "Positive": "😊",
        "Negative": "😞",
        "Neutral": "😐"
    }

    # Add TextBlob results
    table.add_row(["TextBlob", f"{textblob_result[0]} {sentiment_emoji.get(textblob_result[0], '')}",
                   f"{textblob_result[1]:.4f}", f"Subjectivity: {textblob_result[2]:.4f}"])
    # Add VADER results
    table.add_row(["VADER", f"{vader_result[0]} {sentiment_emoji.get(vader_result[0], '')}", 
                   f"{vader_result[1]:.4f}", ""])
    # Add HuggingFace results
    huggingface_score = f"{huggingface_result[1]:.4f}" if huggingface_result[1] else "N/A"
    table.add_row(["Hugging Face", f"{huggingface_result[0]} {sentiment_emoji.get(huggingface_result[0], '')}", 
                   huggingface_score, ""])

    print(table)


def main():
    """
    Main function to drive the sentiment analysis tool.
    """
    os.system("cls" if os.name == "nt" else "clear")
    print_slow(colored("Welcome to the Advanced Sentiment Analysis Tool! 🚀", 'blue', attrs=['bold', 'underline']))
    print_slow("Choose input method: 1 for text, 2 for voice 🎙️")

    choice = input("Enter your choice: ")
    text = None

    if choice == "1":
        text = input("Enter the text for sentiment analysis: ")
    elif choice == "2":
        text = capture_voice_input()
        if not text:
            print(colored("❌ No voice input detected. Exiting...", 'red'))
            return
    else:
        print(colored("❌ Invalid choice. Exiting...", 'red'))
        return

    # Translate and analyze text
    text = translate_text(text)
    print_slow(colored(f"\nTranslated Text (if applicable): {text} 🌍", 'yellow'))

    textblob_result = analyze_textblob(text)
    vader_result = analyze_vader(text)
    huggingface_result = analyze_huggingface(text)

    # Extract insights and display results
    keywords = extract_keywords(text)
    display_results(text, textblob_result, vader_result, huggingface_result, keywords)

    # Visualizations
    plot_sentiment_scores(textblob_result[1], vader_result[1], huggingface_result[1])
    display_wordcloud(keywords, sentiment=textblob_result[0].lower())


if __name__ == "__main__":
    main()
