from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from selenium.webdriver.common.action_chains import ActionChains
import time
import random

def setup_driver():
    """Setup Chrome driver with enhanced anti-detection options and robots.txt compliance"""
    options = webdriver.ChromeOptions()
    
    # Anti-detection measures
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-web-security')
    options.add_argument('--allow-running-insecure-content')
    options.add_argument('--disable-extensions')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # Set a realistic user agent (NOT Googlebot or other crawlers)
    # Using a regular browser user agent to avoid being treated as a bot
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Additional anti-detection
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(options=options)
    
    # Execute script to hide webdriver property
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
    driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})")
    
    # Set window size to look more human
    driver.set_window_size(1366, 768)
    
    return driver

def human_like_delay():
    """Add random human-like delays - respecting robots.txt crawl-delay of 1 second minimum"""
    time.sleep(random.uniform(1.2, 3.5))  # Always at least 1.2 seconds

def check_url_allowed(url):
    """Check if URL is allowed based on robots.txt patterns"""
    disallowed_patterns = [
        '/cart/', '/checkout/', '/buyer/login/otp', '/user/', '/me/', '/order/',
        '/daily_discover/', '/mall/just-for-you/', '/from_same_shop/',
        '/you_may_also_like/', '/find_similar_products/', '/top_products',
        '/search_user', '/addon-deal-selection/', '/bundle-deal/'
    ]
    
    for pattern in disallowed_patterns:
        if pattern in url:
            return False, f"URL contains disallowed pattern: {pattern}"
    
    # Check for disallowed parameters
    disallowed_params = ['sp_atk=', '__classic__=1', 'utm_source', 'searchPrefill']
    for param in disallowed_params:
        if param in url:
            return False, f"URL contains disallowed parameter: {param}"
    
    return True, "URL is allowed"

def check_for_error_page(driver):
    """Check if we hit an error page and handle it"""
    try:
        # Check for "Halaman Tidak Tersedia" error
        error_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Halaman Tidak Tersedia')]")
        if error_elements:
            print("Error page detected. Trying to return to main page...")
            
            # Try clicking "Kembali ke Halaman Utama" button
            try:
                return_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Kembali ke Halaman Utama')]")
                return_button.click()
                time.sleep(3)
                return True
            except:
                # If button not found, go to main Shopee page
                driver.get("https://shopee.co.id")
                time.sleep(3)
                return True
        return False
    except Exception as e:
        print(f"Error checking for error page: {e}")
        return False

def click_pagination_buttons(driver, url):
    """Click buttons 1, 2, 3, 4, 5, 6 in sequence with robots.txt compliance"""
    try:
        # Check if URL is allowed by robots.txt
        allowed, message = check_url_allowed(url)
        if not allowed:
            print(f"URL not allowed by robots.txt: {message}")
            return False
        
        print(f"Navigating to: {url}")
        driver.get(url)
        
        # Respect crawl-delay from robots.txt
        human_like_delay()
        
        # Check for error page
        if check_for_error_page(driver):
            print("Error page encountered, please manually navigate to the correct page")
            return False
        
        wait = WebDriverWait(driver, 15)
        
        # Wait for page to fully load
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        except:
            print("Page failed to load properly")
            return False
        
        print("Page loaded successfully. Looking for pagination buttons...")
        
        # Look for pagination container first
        pagination_selectors = [
            ".shopee-page-controller",
            "[class*='pagination']",
            "[class*='page-controller']",
            "nav[role='navigation']"
        ]
        
        pagination_found = False
        for selector in pagination_selectors:
            try:
                pagination = driver.find_element(By.CSS_SELECTOR, selector)
                print(f"Found pagination container with selector: {selector}")
                pagination_found = True
                break
            except:
                continue
        
        if not pagination_found:
            print("No pagination container found. This might not be a paginated page.")
            return False
        
        # Method 1: Click by button text content
        for i in range(1, 7):
            try:
                print(f"Looking for button {i}...")
                
                # Check for error page before each action
                if check_for_error_page(driver):
                    print("Error page detected during clicking process")
                    return False
                
                # Multiple selectors to try for pagination buttons
                selectors = [
                    f'//button[text()="{i}"]',
                    f'//button[contains(text(), "{i}")]',
                    f'//button[@aria-label="{i}"]',
                    f'//a[text()="{i}"]',
                    f'//a[contains(text(), "{i}")]',
                    f'//span[text()="{i}"]/parent::button',
                    f'//span[text()="{i}"]/parent::a',
                    f'.shopee-page-controller button:contains("{i}")',
                    f'.shopee-page-controller a:contains("{i}")'
                ]
                
                button = None
                used_selector = None
                
                for selector in selectors:
                    try:
                        if selector.startswith('//'):
                            # XPath selector
                            button = wait.until(
                                EC.element_to_be_clickable((By.XPATH, selector))
                            )
                        else:
                            # CSS selector
                            button = wait.until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                            )
                        used_selector = selector
                        break
                    except TimeoutException:
                        continue
                
                if not button:
                    print(f"Button {i} not found with any selector")
                    continue
                
                print(f"Found button {i} using selector: {used_selector}")
                
                # Scroll to button with human-like movement
                actions = ActionChains(driver)
                actions.move_to_element(button).perform()
                
                # Small delay before clicking
                time.sleep(random.uniform(0.5, 1.0))
                
                # Try clicking with different methods
                click_success = False
                
                # Method 1: Regular click
                try:
                    button.click()
                    click_success = True
                    print(f"Successfully clicked button {i} (regular click)")
                except ElementClickInterceptedException:
                    print(f"Button {i} click intercepted, trying alternative methods")
                
                # Method 2: JavaScript click
                if not click_success:
                    try:
                        driver.execute_script("arguments[0].click();", button)
                        click_success = True
                        print(f"Successfully clicked button {i} (JavaScript click)")
                    except Exception as e:
                        print(f"JavaScript click failed for button {i}: {e}")
                
                # Method 3: ActionChains click
                if not click_success:
                    try:
                        actions = ActionChains(driver)
                        actions.click(button).perform()
                        click_success = True
                        print(f"Successfully clicked button {i} (ActionChains click)")
                    except Exception as e:
                        print(f"ActionChains click failed for button {i}: {e}")
                
                if not click_success:
                    print(f"Failed to click button {i} with all methods")
                    continue
                
                # Human-like delay after clicking (respecting robots.txt crawl-delay)
                human_like_delay()
                
                # Check if we got redirected to error page
                if check_for_error_page(driver):
                    print("Redirected to error page after clicking")
                    return False
                
                print(f"Completed clicking button {i}, waiting before next action...")
                
            except Exception as e:
                print(f"Error with button {i}: {e}")
                continue
    
    except Exception as e:
        print(f"General error: {e}")
        return False
    
    return True

def click_pagination_buttons_by_class(driver, url):
    """Alternative method: Click by class name and position"""
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        
        # Wait for pagination container to load
        pagination_nav = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "shopee-page-controller"))
        )
        
        # Find all numbered buttons (excluding arrow buttons)
        numbered_buttons = driver.find_elements(
            By.CSS_SELECTOR, 
            'button.shopee-button-no-outline:not(.shopee-icon-button)'
        )
        
        # Filter buttons that contain numbers 1-6
        target_buttons = []
        for button in numbered_buttons:
            text = button.text.strip()
            if text.isdigit() and 1 <= int(text) <= 6:
                target_buttons.append((int(text), button))
        
        # Sort by number and click in sequence
        target_buttons.sort(key=lambda x: x[0])
        
        for number, button in target_buttons:
            try:
                print(f"Clicking button {number}...")
                
                # Scroll to button
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                time.sleep(0.5)
                
                # Check if button is clickable
                if button.is_enabled():
                    button.click()
                    time.sleep(2)
                    print(f"Successfully clicked button {number}")
                else:
                    print(f"Button {number} is disabled")
                    
            except Exception as e:
                print(f"Failed to click button {number}: {e}")
    
    except Exception as e:
        print(f"Error: {e}")

def click_specific_shopee_buttons(driver, url):
    """Method specifically for the HTML structure shown"""
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        
        # CSS selectors for the specific buttons
        button_selectors = [
            'button.shopee-button-no-outline:nth-of-type(2)',  # Button 1
            'button.shopee-button-no-outline:nth-of-type(3)',  # Button 2
            'button.shopee-button-solid.shopee-button-solid--primary',  # Button 3 (active)
            'button.shopee-button-no-outline:nth-of-type(5)',  # Button 4
            'button.shopee-button-no-outline:nth-of-type(6)',  # Button 5
            'button.shopee-button-no-outline:nth-of-type(7)',  # Button 6
        ]
        
        for i, selector in enumerate(button_selectors, 1):
            try:
                print(f"Clicking button {i}...")
                
                button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                
                # Scroll and click
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                time.sleep(0.5)
                button.click()
                time.sleep(2)
                
                print(f"Successfully clicked button {i}")
                
            except Exception as e:
                print(f"Failed to click button {i}: {e}")
    
    except Exception as e:
        print(f"Error: {e}")

# Main execution with robots.txt compliance
if __name__ == "__main__":
    # Initialize driver
    driver = setup_driver()
    
    try:
        print("=== Shopee Button Auto-Clicker (Robots.txt Compliant) ===")
        print("This script respects Shopee's robots.txt crawl-delay of 1+ seconds")
        print()
        
        # Get URL from user
        shopee_url = input("Enter Shopee product page URL with pagination: ").strip()
        
        if not shopee_url:
            print("No URL provided. Please provide a valid Shopee product page URL.")
            exit()
        
        # Validate URL
        if not shopee_url.startswith(('http://', 'https://')):
            shopee_url = 'https://' + shopee_url
        
        if 'shopee.co.id' not in shopee_url and 'shopee.com' not in shopee_url:
            print("Warning: This doesn't appear to be a Shopee URL")
        
        # Check robots.txt compliance
        allowed, message = check_url_allowed(shopee_url)
        if not allowed:
            print(f"Error: {message}")
            print("Please use a different URL that complies with robots.txt")
            exit()
        
        print(f"URL is robots.txt compliant: {message}")
        
        # First, navigate to Shopee main page to establish session
        print("Establishing session with Shopee...")
        driver.get("https://shopee.co.id")
        human_like_delay()
        
        # Check if we can access Shopee
        if check_for_error_page(driver):
            print("Cannot access Shopee. Possible reasons:")
            print("1. Network connectivity issues")
            print("2. Shopee is blocking automated access")
            print("3. Your IP might be temporarily blocked")
            print("\nSuggestions:")
            print("- Try again later")
            print("- Use a VPN")
            print("- Run the script manually after logging in")
        else:
            print("Successfully accessed Shopee")
            
            # Navigate to target page
            print(f"Navigating to target page: {shopee_url}")
            success = click_pagination_buttons(driver, shopee_url)
            
            if success:
                print("✅ Successfully completed pagination button clicking!")
            else:
                print("❌ Failed to complete button clicking process")
                print("\nTroubleshooting suggestions:")
                print("1. Ensure the page has pagination buttons numbered 1-6")
                print("2. Check if you need to log in first")
                print("3. Verify the URL is a product listing page with reviews/ratings")
                print("4. Try manually clicking one button first to see the page structure")
        
        # Manual intervention option
        print("\n=== Manual Intervention Available ===")
        print("The browser will stay open for manual navigation if needed")
        print("You can:")
        print("1. Manually navigate to the correct page")
        print("2. Log in if required")
        print("3. Test clicking buttons manually")
        
        input("Press Enter when done or to close browser...")
        
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("This might be due to:")
        print("1. Shopee's anti-bot protection")
        print("2. Network issues") 
        print("3. Page structure changes")
        print("4. Invalid URL format")
    finally:
        # Clean up
        try:
            driver.quit()
            print("Browser closed successfully")
        except:
            print("Browser cleanup completed")