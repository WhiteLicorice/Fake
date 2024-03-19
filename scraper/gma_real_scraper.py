import time
import requests
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

base_url = "https://www.gmanetwork.com/news/balitambayan/balita/"
num_articles_to_scrape = 1600  # Set the desired number of articles to scrape

driver = webdriver.Chrome()  # Use Selenium to initiate a browser session
driver.get(base_url)

csv_file_path = 'gma_real_news_2k.csv'
with open(csv_file_path, 'w', encoding='utf-8-sig', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['label', 'article'])

    articles_scraped = 0
    last_height = driver.execute_script("return document.body.scrollHeight")
    scraped_urls = set()  # Set to store scraped article URLs

    while articles_scraped < num_articles_to_scrape:
        # Scroll down first
        driver.find_element("tag name", 'body').send_keys(Keys.END)

        # Use WebDriverWait to wait for the new content to load
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'stories')))
        except TimeoutException:
            print("Timeout occurred while waiting for new content to load.")
            break

        time.sleep(2)  # Introducing a short pause for stability

        updated_page_source = driver.page_source
        soup_base = BeautifulSoup(updated_page_source, 'html.parser')

        news_title_divs = soup_base.find_all('div', class_='stories')

        for news_title_div in news_title_divs:
            try:
                anchor_tag = news_title_div.find('a')
                if anchor_tag:
                    article_link = f"https://www.gmanetwork.com{anchor_tag['href']}"
                    if article_link in scraped_urls:  # Check if article URL is already scraped
                        continue
                    else:
                        scraped_urls.add(article_link)

                    response_article = requests.get(article_link, allow_redirects=True)

                    if response_article.status_code == 200:
                        soup_article = BeautifulSoup(response_article.text, 'html.parser')
                        article_content_div = soup_article.find('div', class_='article-body')

                        if article_content_div:
                            # Extract the text content of the article
                            paragraphs = article_content_div.find_all('p')
                            content_text = ' '.join([p.get_text().strip() for p in paragraphs])
                            csv_writer.writerow(['1', content_text])
                            print(f"Scraped Article {articles_scraped + 1}")
                            articles_scraped += 1

                            # Check if the desired number of articles is reached
                            if articles_scraped >= num_articles_to_scrape:
                                break
                        else:
                            print(f"Skipped article without content: {article_link}")
                    else:
                        print(f"Failed to retrieve the individual article. Status code: {response_article.status_code}")

            except Exception as e:
                print(f"An error occurred: {e}")

        # new_height = driver.execute_script("return document.body.scrollHeight")

        # if new_height == last_height:
        #     break

        # last_height = new_height

print(f"Scraped {articles_scraped} articles. Data saved to {csv_file_path}")
driver.quit()
