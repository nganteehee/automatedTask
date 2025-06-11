import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException,WebDriverException
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
class ShopeeFoodScraper:
    def __init__(self, headless=True, wait_time=15):
       
       self.wait_time = wait_time
       self.headless = headless
       self.driver = None
       self.setup_driver()
    def setup_driver(self):
        """Setup Chrome driver with optimized options for Shopee Food"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # Optimized options for JavaScript-heavy sites
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--disable-images')  # Faster loading
        chrome_options.add_argument('--disable-javascript')  # Remove this if JS is needed
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Memory optimization
        chrome_options.add_argument('--memory-pressure-off')
        chrome_options.add_argument('--max_old_space_size=4096')
        
        # Set user agent
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        try:
            self.driver = webdriver.Chrome(options = chrome_options)
            #set page load time out
            self.driver.set_page_load_timeout(30)
            #set time for element to be found
            self.driver.implicitly_wait(10)
            #set the navigator.webdriver to undefined
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print('Webdriver initialzied successfully')
        except Exception as e:
            print(f"Error setting up WebDriver: {e}")
            raise
    def check_driver_health(self):
        """Check if the WebDriver is still alive and responsive"""
        try:
            current_url = self.driver.current_url
            return True
        except WebDriverException as e:
            print(f'unexpected webdriver exception {e}')
            return False
        except Exception as e:
            print(f'unexpected error exception {e}')
        
    def restart_driver(self):
        """Restart the WebDriver if it crashes"""
        print("Restarting WebDriver...")
        try: 
            if self.driver:
                self.driver.quit()
        except: 
            pass
        self.setup_driver()    
    def safe_navigate(self, url, retries=3):
        """Safely navigate to a URL with retry logic"""
        
        for attempt in range(retries):
            try:
                if not self.check_driver_health():
                    print(f'attemp to restart driver .... Attempt {attempt+1}')
                    self.restart_driver()
                print(f'Navigate to {url} (attempt {attempt + 1})')    
                self.driver.get(url)
                return True
            except WebDriverException as e:
                print(f'WebDriver exception on attempt {attempt + 1}')
                if attempt < retries-1:
                    print('Restarting driver...')
                    self.setup_driver()
                    time.sleep(5)
                else:
                    print(f'giving up on url {url}')
                    return False
            except Exception as e:
                print(f'Unexpected Error at attempt {attempt + 1}')
                if attempt < retries-1:
                    print('Restarting driver...')
                    self.setup_driver()
                    time.sleep(5)
                else:
                    print(f'giving up on url {url}')
                    return False
            return False

    
    def wait_for_content_load(self, url):
        """Navigate to URL and wait for content to load with retry logic"""
        if not self.safe_navigate(url):
            return False
        
        try:
            # Wait for the page to load - looking for common Shopee Food elements
            possible_selectors = [
                '[data-testid="restaurant-item"]',
                '.restaurant-item',
                '.item-restaurant',
                '.shop-card',
                '[class*="restaurant"]',
                '[class*="shop"]',
                '.list-restaurant',
                '#main-content',
                '.main-content',
                '[class="row"]'
            ]
            content_load = False
            for selector in possible_selectors:
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR,selector ))
                    )
                    print(f'content loaded with selector {selector}')
                    content_load = True
                    break
                except TimeoutException:
                    continue
            time.sleep(3)
            return True
        except Exception as e:
            print(f'Exception {e} when loading...')
            return False
           
    
    def extract_restaurant_data(self):
        """Extract restaurant data from the loaded page"""
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        restaurants = []
        restaurant_elements = []
        restaurant_selectors = [
                '[data-testid="restaurant-item"]',
                '.restaurant-item',
                '.shop-card',
                '[class*="restaurant"]',
                '[class*="shop-card"]',
                'div[class*="ShopCard"]'
            ]
        for selector in restaurant_selectors:
            element = soup.select(selector)
            if element:
                print(f'found {len(element)} restaurant by {selector}')
                restaurant_elements = element
                break
        if not restaurant_elements:
            # Fallback: try to find any clickable food-related elements
            restaurant_elements = soup.find_all('div', class_=lambda x: x and any(
                keyword in x.lower() for keyword in ['shop', 'restaurant', 'food', 'store']
            ))
            print(f"Fallback: Found {len(restaurant_elements)} potential restaurant elements")
        for restaurant in restaurant_elements:
            restaurant_data = self.extract_single_restaurant(restaurant)
            if restaurant_data.get('name') and restaurant_data:
                restaurants.append(restaurant_data)
        return restaurants
    def extract_single_restaurant(self, element):
        """Extract data from a single restaurant element"""
        data = {
            'name': '',
            'address': '',
            'image_url': '',
            'promo': '',
            'href': ''
        }
        
        try:
            # Restaurant name - try multiple selectors
            name_selectors = [
            'h3', 'h2', 'h4',
            '[class*="name"]',
            '[class*="title"]',
            '.shop-name',
            '.restaurant-name'
            ]
            for selector in name_selectors:
                name = element.select_one(selector)
                if name and name.get_text().strip():
                    data['name'] = name.get_text().strip()
                    break
            #get address
            address_selector = ['[class="address-res"]']
            for selector in address_selector:
                address = element.select_one(selector)
                if address and address.get_text().strip():
                    data['address'] = address.get_text().strip()
                    break
            # Image
            img_element = element.select_one('img')
            if img_element:
                data['image_url'] = img_element.get('src', '') or img_element.get('data-src', '')
            
            # Promo/discount
            promo_selectors = [
                '[class*="promo"]',
                '[class*="discount"]',
                '[class*="offer"]',
                '.badge'
            ]
            
            for selector in promo_selectors:
                promo_element = element.select_one(selector)
                if promo_element:
                    data['promo'] = promo_element.get_text().strip()
                    break
            #link href
            href_selectors = [
                '.item-content',
                'a[href*="/ho-chi-minh/"]',
                'a[target="_blank"]'
            ]
            
            for selector in href_selectors:
                href_element = element.select_one(selector)
                if href_element and href_element.get('href'):
                    href = href_element.get('href')
                    # Make it a full URL if it's relative
                    if href.startswith('/'):
                        href = f"https://shopeefood.vn{href}"
                    data['href'] = href
                    break
            return data
        
        except Exception as e:
            print(f'Error trying extracting data from single restaurant')
            return data
            
    
    def scroll_to_load_more(self, max_scrolls=5):
        """Scroll down to load more restaurants (for infinite scroll)"""
        try:
            for croll in range(max_scrolls):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                print(f'scroll down {croll + 1}/{max_scrolls} times')
        except Exception as e:
            print(f'Error when crolling {e}')
    def extract_menu_restaurant(self, restaurant):
        url = restaurant.get('href')
        if not url:
            print(f"No URL found for restaurant: {restaurant.get('name', 'Unknown')}")
            restaurant['menu'] = {}
            return restaurant
        if not self.check_driver_health():
            print("Driver unhealthy, restarting before menu extraction...")
            self.restart_driver()
        if not self.wait_for_content_load(url):
            print(f"Failed to load restaurant page: {url}")
            restaurant['menu'] = {}
            return restaurant
        
        # Wait for menu items to load
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[class="row"], .menu-item, [class*="item"], [class*="product"]'))
            )
        except TimeoutException:
            print("Menu items not found, continuing with available content...")        
        # self.wait_for_content_load(url)
        # self.driver.get(url)
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        menu = soup.select('[class = "row"]')
        dct = dict()
        for item in menu:
            try:
                # print(item)
                # menu[0].select_one('[class*="name"]').get_text()
                name = item.select_one('[class*="name"]').get_text()
                price = item.select_one('[class = current-price]').get_text()
                dct[name] = price
            except:
                pass
        restaurant['menu'] = dct



        return restaurant

    def click_pagination_next(self, max_pages=10):
        """
        Automatically click through pagination links
        
        Args:
            max_pages: Maximum number of pages to navigate through
        """
        restaurants_all_page = []
        current_page = 1
        while current_page < max_pages:
            print(f'currently on page {current_page}')
            print(f'extracting data...')
            restaurant_data = self.extract_restaurant_data()
            restaurants_all_page.append(restaurant_data)
            print(f"Successfully extracted {len(restaurant_data)} restaurants")
            time.sleep(2)
            try:
                #Find by href pattern if it changes
                next_link = self.driver.find_element(By.XPATH, f"//a[@href='#' and text()='{current_page + 1}']")
                self.driver.execute_script("arguments[0].scrollIntoView();", next_link)
                time.sleep(3)
                current_page += 1
            except (NoSuchElementException, TimeoutException):
                print(f"No more pages found after page {current_page}")
                break
            except Exception as e:
                print(f"Error clicking pagination: {e}")
                break
        #handle nested list
        if restaurants_all_page and isinstance(restaurants_all_page[0], list):
            # Use list comprehension instead of np.squeeze for better control
            flattened_restaurants = []
            for page_restaurants in restaurants_all_page:
                if isinstance(page_restaurants, list):
                    flattened_restaurants.extend(page_restaurants)
                else:
                    flattened_restaurants.append(page_restaurants)
            return flattened_restaurants
        else:
            return restaurants_all_page
        
    def scrape_shopee_food(self, url, scroll_for_more=True):
        """Main scraping function"""
        detail_restaurent = []
        try: 

            if not self.wait_for_content_load(url):
                print(f'fail to load content...')
                return []
            if scroll_for_more:
                self.scroll_to_load_more()
                print(f'scroll to load more content')
            data = self.click_pagination_next()
            
            for restaurant in data:
                detail = self.extract_menu_restaurant(restaurant)
                detail_restaurent.append(detail)
            return detail_restaurent
        except Exception as e:
            print(f'Error while scrapeing {e}')
            return detail_restaurent
    
    def save_data(self, data, filename_prefix="shopee_food"):
        """Save scraped data to files"""
        if not data:
            print(f'No data to save')
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        #save to json
        json_name = f'{filename_prefix}_{timestamp}.json'
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
            print(f'data save to {json_name}')
        #save to csv
        csv_name = f'{filename_prefix}_{timestamp}.csv'
        df = pd.DataFrame(data)
        df.to_csv(csv_name, index=False, encoding='utf-8')
        print(f"Data saved to {csv_name}")

    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()

def main():
    """Example usage"""
    # url = "https://shopeefood.vn/ho-chi-minh/food"
    path = 'https://shopeefood.vn/ho-chi-minh/danh-sach-dia-diem-giao-tan-noi?q='
    keywords = ['trà sữa', 'gà rán', 'nước ép']
    # keywords = ['trà sữa']
    urls = []
    for keyword in keywords:
        urls.append(path + keyword)
    # Create scraper instance
    scraper = ShopeeFoodScraper(headless=False)  # Set to True to run in background
    
    try:
        # Scrape the data
        for url in urls: 
            restaurants = scraper.scrape_shopee_food(url, scroll_for_more=True)
            
            if restaurants:
                
                # Print first few results
                print("\nFirst 3 restaurants found:")
                print(restaurants)
                # for i, restaurant in enumerate(restaurants[:3], 1):
                #     print(f"\n{i}. {restaurant.get('name', 'N/A')}")
                #     print(f"   Adress: {restaurant.get('adress', 'N/A')}")
                #     print(f"   Link: {restaurant.get('href', 'N/A')}")
                
                # Save the data
                scraper.save_data(restaurants)

            else:
                print("No restaurant data found. The site structure might have changed.")
                print("You may need to update the CSS selectors in the code.")
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main()