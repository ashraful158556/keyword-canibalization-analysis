import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import logging

# লগিং সেটআপ
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ইউআরএল লোড করা
def load_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# কনটেন্ট স্ক্র্যাপিং ফাংশন
def fetch_page_content(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        # শুধু <p>, <h1-h3>, <title> ট্যাগ নেয়া
        texts = soup.find_all(['p', 'h1', 'h2', 'h3', 'title'])
        content = ' '.join([tag.get_text(strip=True) for tag in texts])
        return content
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return ""

# মূল স্ক্রিপ্ট
def main():
    urls = load_urls('urls.txt')
    logging.info(f"{len(urls)} URLs loaded. Fetching content...")

    documents = []
    for url in urls:
        content = fetch_page_content(url)
        documents.append(content)

    logging.info("Similarity analysis using TF-IDF...")

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    threshold = 0.3  # similarity threshold, চাইলে পরিবর্তন করুন

    cannibalized_pairs = []

    for i in range(len(urls)):
        for j in range(i + 1, len(urls)):
            similarity_score = similarity_matrix[i][j]
            if similarity_score > threshold:
                cannibalized_pairs.append((urls[i], urls[j], round(similarity_score, 3)))

    logging.info(f"Total found {len(cannibalized_pairs)} cannibalized pages")

    # রিপোর্ট তৈরি করা
    with open('cannibalization_report.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Page 1', 'Page 2', 'Similarity Score'])
        writer.writerows(cannibalized_pairs)

    logging.info("✅ Report: Saved in cannibalization_report.csv file.")

if __name__ == '__main__':
    main()
