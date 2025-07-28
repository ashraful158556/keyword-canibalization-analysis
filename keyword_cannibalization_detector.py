import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# কনটেন্ট + ট্যাগ আলাদা করে আনবে
def fetch_page_details(url):
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.get_text(strip=True) if soup.title else ''
        h1 = ' | '.join([tag.get_text(strip=True) for tag in soup.find_all('h1')])
        h2 = ' | '.join([tag.get_text(strip=True) for tag in soup.find_all('h2')])
        h3 = ' | '.join([tag.get_text(strip=True) for tag in soup.find_all('h3')])
        p_tags = ' '.join([tag.get_text(strip=True) for tag in soup.find_all('p')])

        full_content = f"{title} {h1} {h2} {h3} {p_tags}"

        return {
            'url': url,
            'title': title,
            'h1': h1,
            'h2': h2,
            'h3': h3,
            'content': full_content
        }
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return {
            'url': url,
            'title': '',
            'h1': '',
            'h2': '',
            'h3': '',
            'content': ''
        }

def main():
    urls = load_urls('urls.txt')
    logging.info(f"{len(urls)} URLs loaded. Fetching content...")

    pages = [fetch_page_details(url) for url in urls]
    documents = [page['content'] for page in pages]

    logging.info("Similarity analysis using TF-IDF...")

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    threshold = 0.3
    cannibalized_pairs = []

    for i in range(len(pages)):
        for j in range(i + 1, len(pages)):
            similarity_score = similarity_matrix[i][j]
            if similarity_score > threshold:
                cannibalized_pairs.append({
                    'url_1': pages[i]['url'],
                    'title_1': pages[i]['title'],
                    'h1_1': pages[i]['h1'],
                    'url_2': pages[j]['url'],
                    'title_2': pages[j]['title'],
                    'h1_2': pages[j]['h1'],
                    'similarity': round(similarity_score, 3)
                })

    logging.info(f"Total found {len(cannibalized_pairs)} cannibalized pages")

    # রিপোর্ট ফাইল সংরক্ষণ
    with open('cannibalization_report.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Page 1 URL', 'Page 1 Title', 'Page 1 H1', 
                         'Page 2 URL', 'Page 2 Title', 'Page 2 H1', 
                         'Similarity Score'])
        for row in cannibalized_pairs:
            writer.writerow([
                row['url_1'], row['title_1'], row['h1_1'],
                row['url_2'], row['title_2'], row['h1_2'],
                row['similarity']
            ])

    logging.info("✅ Report saved as 'cannibalization_report.csv'")

if __name__ == '__main__':
    main()
