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
            'content': full_content
        }
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return {
            'url': url,
            'title': '',
            'h1': '',
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

    threshold = 0.7
    grouped_results = []

    for i in range(len(pages)):
        main_page = pages[i]
        for j in range(len(pages)):
            if i == j:
                continue
            similarity_score = similarity_matrix[i][j]
            if similarity_score > threshold:
                grouped_results.append({
                    'main_url': main_page['url'],
                    'matched_url': pages[j]['url'],
                    'main_title': main_page['title'],
                    'matched_title': pages[j]['title'],
                    'main_h1': main_page['h1'],
                    'matched_h1': pages[j]['h1'],
                    'similarity': round(similarity_score, 3)
                })

    logging.info(f"Total matched pairs found: {len(grouped_results)}")

    # রিপোর্ট সংরক্ষণ
    with open('cannibalization_group_report.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Group Main URL', 'Matched URL',
            'Group Main Title', 'Matched Title',
            'Group Main H1', 'Matched H1',
            'Similarity Score'
        ])
        for row in grouped_results:
            writer.writerow([
                row['main_url'], row['matched_url'],
                row['main_title'], row['matched_title'],
                row['main_h1'], row['matched_h1'],
                row['similarity']
            ])

    logging.info("✅ Grouped report saved as 'cannibalization_group_report.csv'")

if __name__ == '__main__':
    main()
