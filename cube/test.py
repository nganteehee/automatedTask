import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd

class ExistingBrowserScraper:
    def __init__(self):
        self.driver = None
    
    # METHOD 1: Connect to existing Chrome with remote debugging
    def connect_to_existing_chrome(self, debug_port=9222):
        """
        Connect to an existing Chrome browser with remote debugging enabled.
        
        Steps to use:
        1. Close all Chrome instances
        2. Open Chrome with: chrome --remote-debugging-port=9222
        3. Navigate to your desired page manually
        4. Run this method to connect
        """
        try:
            chrome_options = Options()
            chrome_options.add_experimental_option("debuggerAddress", f"127.0.0.1:{debug_port}")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            print(f"Successfully connected to existing Chrome session")
            print(f"Current URL: {self.driver.current_url}")
            print(f"Page Title: {self.driver.title}")
            return True
            
        except Exception as e:
            print(f"Error connecting to existing Chrome: {e}")
            print("Make sure Chrome is running with --remote-debugging-port=9222")
            return False
    def get_current_page_data(self):
        """Extract data from currently loaded page"""
        if not self.driver:
            print("No driver connection available")
            return None
        
        try:
            current_url = self.driver.current_url
            page_title = self.driver.title
            page_source = self.driver.page_source
            
            print(f"Scraping data from: {current_url}")
            print(f"Page title: {page_title}")
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Generic data extraction
            # data = {
            #     'name': '',
            #     'sold': '',
            #     'price': '', 
            #     'rating': ''
            # }
            list_product = []
            product_element = '[class="buTCk"]'
            elements = soup.select(product_element)
            for element in elements:
                data = {
                'name': '',
                'sold': '',
                'price': '', 
                'rating': ''
            }
                data['price'] = element.select_one('[class="aBrP0"]').get_text()
                data['name'] = element.select_one('[class="RfADt"]').get_text()
                data['sold'] = element.select_one('[class="_6uN7R"]').get_text()
                print(data)
                list_product.append(data)
            # Extract common elements
            # data['content']['headings'] = [h.get('title').strip() for h in soup.find_all(['h1', 'h2', 'h3'])]
            # data['content']['links'] = [{'text': a.get_text().strip(), 'href': a.get('href')} 
            #                           for a in soup.find_all('a', href=True)]
            # data['content']['images'] = [{'alt': img.get('alt', ''), 'src': img.get('src')} 
            #                            for img in soup.find_all('img')]
            
            return list_product
            
        except Exception as e:
            print(f"Error extracting data: {e}")
            return None
    def extract_shopee_product_data(self):
        """Extract product data from Shopee page"""
        if not self.driver:
            print("No driver connection available")
            return []
        
        try:
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            products = []
            
            # Common Shopee product selectors
            product_selectors = [
                # '[class*="shopee-product-rating"]',
                # '[class="shopee-product-rating__time"]',
                '[class="item"]'

            ]
            
            product_elements = []
            for selector in product_selectors:
                elements = soup.select(selector)
                if elements:
                    print(f"Found {len(elements)} products with selector: {selector}")
                    product_elements = elements
                    print(product_elements)
                    break
            
            if not product_elements:
                print("No products found, trying alternative approach...")
                # Fallback: look for any divs that might contain products
                product_elements = soup.find_all('div', class_=lambda x: x and any(
                    keyword in x.lower() for keyword in ['item', 'product', 'card']
                ))
            
            for element in product_elements:
                product_data = self.extract_single_product(element)
                # if element.get_text():
                products.append(product_data)
            
            print(f"Successfully extracted {len(products)} products")
            return products
            
        except Exception as e:
            print(f"Error extracting Shopee products: {e}")
            return []
    def extract_single_product(self, element):
        """Extract data from a single product element"""
        data = {
            'review_time': '',
            'content': '',
            'rating': '',
            'variaty': '',
            # 'rating': '',
            # 'sold': '',
            # 'image_url': '',
            # 'product_url': ''
        }
        
        try:
            # Product name
            time_selectors = [
            'span.title.right font',
            '.title.right font',
            'span.title font',
            '.item .title.right font',  # Sometimes name is in title attribute
            '[class="container-star starCtn left"]'
            ]
            
            for selector in time_selectors:
                name_elem = element.select_one(selector)
                if name_elem:
                    data['review_time'] = name_elem.get_text().strip()
                    break
            
            # Price
            content_selectors = [
                'div.content font',
                '.content font',
            '.item-content .content font',
            'div.content font[data-spm-anchor-id*="ratings_reviews"]',
            '[class="content"]'
            ]
            
            for selector in content_selectors:
                price_elem = element.select_one(selector)
                if price_elem:
                    data['content'] = price_elem.get_text().strip()
                    break
            
            # # Image
            # img_elem = element.select_one('img')
            # if img_elem:
            #     data['image_url'] = img_elem.get('src') or img_elem.get('data-src')
            
            # # Product URL
            # link_elem = element.select_one('a')
            # if link_elem and link_elem.get('href'):
            #     href = link_elem.get('href')
            #     if href.startswith('/'):
            #         href = f"https://shopee.co.id{href}"
            #     data['product_url'] = href
            
            # # Rating and sold count (if available)
            # rating_elem = element.select_one('[class*="rating"]')
            # if rating_elem:
            #     data['rating'] = rating_elem.get_text().strip()
            
            # sold_elem = element.select_one('[class*="sold"]')
            # if sold_elem:
            #     data['sold'] = sold_elem.get_text().strip()
            
            print(data)
            return data
            
        except Exception as e:
            print(f"Error extracting single product: {e}")
            return data
    
    # METHOD 5: Interactive scraping with user navigation
    def interactive_scrape(self):
        """Let user navigate manually, then scrape on command"""
        if not self.driver:
            print("No driver connection available")
            return
        
        print("Interactive scraping mode activated!")
        print("Navigate to your desired page manually in the browser.")
        print("Commands:")
        print("  'scrape' - Extract data from current page")
        print("  'shopee' - Extract Shopee product data")
        print("  'url' - Show current URL")
        print("  'quit' - Exit")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'url':
                print(f"Current URL: {self.driver.current_url}")
            elif command == 'scrape':
                data = self.get_current_page_data()
                if data:
                    self.save_data(data, "scraped_page")
            elif command == 'shopee':
                products = self.extract_shopee_product_data()
                if products:
                    self.save_data(products, "shopee_products")
            else:
                print("Unknown command")
    
    def click_pagination_buttons(self, page_numbers):
        """Click buttons 1, 2, 3, 4, 5, 6 in sequence"""
        total_review = []
        try:
            for page_number in range(1, page_numbers):
                # for i, button in enumerate(buttons):
                try:
                    next_button = WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable(
                                (By.CSS_SELECTOR, 'button[class*="next-pagination-item next"]')
                    ))
                    next_button.click()
                    print("✓ Successfully clicked using class names")
                    self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", next_button)
                    time.sleep(20)
                
                # Try to click
                    try:
                    
                        review = self.extract_shopee_product_data()
                        total_review.append(review)
                        time.sleep(20)
                        print(f"Successfully clicked page {page_number}")
                    except Exception as click_error:
                        print(f"Regular click failed: {click_error}")
                        continue
                except Exception as e:
                    print(f"Error examining button {page_number}: {e}")
                    continue
                        
                        # if target_button:
                            # Scroll into view
        #                     self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", target_button)
        #                     time.sleep(20)
                            
        #                     # Try to click
        #                     try:
                                
        #                         review = self.extract_shopee_product_data()
        #                         total_review.append(review)
        #                         target_button.click()
        #                         time.sleep(20)
        #                         print(f"Successfully clicked page {page_number}")
        #                         # return True
        #                     except Exception as click_error:
        #                         print(f"Regular click failed: {click_error}")
        #                         # Try JavaScript click
        #                         self.driver.execute_script("arguments[0].click();", target_button)
        #                         print(f"JavaScript click succeeded for page {page_number}")
        #                         # return True
                                
        #                 else:
        #                     print(f"Could not find clickable button for page {page_number}")
                    
        except Exception as e:
            print(f"Error: {e}")
        
        self.save_data(total_review, 'lzd_detail_review_ESQA_Flawless_Setting_Powder')
        return total_review
    
    def save_data(self, data, filename_prefix):
        """Save data to JSON and CSV files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {json_filename}")
        
        # Save CSV if it's a list of dictionaries
        if isinstance(data, list) and data and isinstance(data[0], dict):
            csv_filename = f"{filename_prefix}_{timestamp}.csv"
            df = pd.DataFrame(data)
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"Data saved to {csv_filename}")
    def close(self):
        """Close the browser connection"""
        if self.driver:
            self.driver.quit()
def example_usage():
    scraper = ExistingBrowserScraper()
    scraper.connect_to_existing_chrome()
    scraper.interactive_scrape()
    scraper.click_pagination_buttons(120)

    scraper.close()
if __name__ == "__main__":
    example_usage()
