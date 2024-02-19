import requests
import time
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

base_url = "https://www.philstar.com/pilipino-star-ngayon"

# Set the number of scrolls and articles to scrape
num_scrolls = 1000 # Set the desired number of scrolls
num_articles_to_scrape = 1000  # Set the desired number of articles to scrape

# Use Selenium to initiate a browser session
driver = webdriver.Chrome()  # You need to have ChromeDriver installed and its path set in your system

# Send a GET request to the base URL
driver.get(base_url)

# Function to scroll down the page
def scroll_down():
    for _ in range(num_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Adjust the sleep duration as needed

# Scroll down to trigger the loading of more content
scroll_down()

# Get the updated HTML content after scrolling
updated_page_source = driver.page_source

# Close the browser
driver.quit()

# Parse the updated HTML content
soup_base = BeautifulSoup(updated_page_source, 'html.parser')

# Find all <div> elements with class "news_title"
news_title_divs = soup_base.find_all('div', class_='news_title')

# Create a CSV file to store the scraped data
csv_file_path = 'real_news.csv'
with open(csv_file_path, 'w', encoding='utf-8-sig', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Label', 'Article'])

    # Extract and print the news articles
    articles_scraped = 0
    for news_title_div in news_title_divs:
        if articles_scraped >= num_articles_to_scrape:
            break

        article_link = news_title_div.find('a')['href']

        # Access the individual article link
        response_article = requests.get(article_link)

        # Check if the request was successful (status code 200) for the individual article
        if response_article.status_code == 200:
            # Parse the HTML content of the individual article page
            soup_article = BeautifulSoup(response_article.text, 'html.parser')

            # Find the article title and content
            article_content = soup_article.find('div', {'id': 'sports_article_writeup', 'class': 'article__writeup'})

            # Check if both title and content are found
            if article_content:
                # Extract the text content of the title and content
                content_text = article_content.get_text().replace('\n', ' ').strip()

                # Write the data to the CSV file
                csv_writer.writerow(['1', content_text])
                print(f"Scraped Article {articles_scraped + 1}")

                articles_scraped += 1
            else:
                print(f"Skipped article without title or content: {article_link}")
        else:
            print(f"Failed to retrieve the individual article. Status code: {response_article.status_code}")

print(f"Scraped {articles_scraped} articles. Data saved to {csv_file_path}")
