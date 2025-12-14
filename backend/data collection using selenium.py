import os
import time
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from queue import Queue
from threading import Lock

class Config:
    BASE_URL = "https://indiankanoon.org"
    INITIAL_SEARCH_URL = "https://indiankanoon.org/search/?addlq=contract&formInput=doctypes%3Asupremecourt"
    PDF_DIR = "Indian_Kanoon_PDFs"
    WAIT_TIME = 20
    PAGE_LOAD_DELAY = 10
    MAX_WORKERS = 1  # Number of parallel browsers

def setup_driver():
    options = Options()
    # options.add_argument('--headless')  # Uncomment for headless mode
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.maximize_window()
    return driver

def wait_and_click(wait, by, value, message=""):
    try:
        element = wait.until(EC.element_to_be_clickable((by, value)))
        print(f"Found element: {message}")
        element.click()
        time.sleep(Config.PAGE_LOAD_DELAY)
        return True
    except Exception as e:
        print(f"Failed to click element {message}: {str(e)}")
        return False

def process_single_document(driver, wait, url, lock):
    try:
        driver.get(url)
        time.sleep(Config.PAGE_LOAD_DELAY)

        # Main-Task: Try Task A first
        # Task A: Click on Component-B in Page-C to download PDF
        success = task_a(driver, wait, lock)
        if not success:
            print("Task A failed, attempting Task B")
            # Task B: Click on Component-A in Page-F, then Component-B in Page-C to download PDF
            success = task_b(driver, wait, lock)
        return success
    except Exception as e:
        print(f"Error processing document {url}: {e}")
        return False

def task_a(driver, wait, lock):
    try:
        # Assume we are on Page-A and have clicked on a link to Page-C
        # Try to find Component-B to download PDF
        if wait_and_click(wait, By.ID, "pdfdoc", "PDF button (Component-B)"):
            return True
        else:
            return False
    except Exception as e:
        print(f"Error in Task A: {e}")
        return False

def task_b(driver, wait, lock):
    try:
        # From Page-A, click on link to Page-F (Component-E)
        if wait_and_click(wait, By.LINK_TEXT, "View Complete document", "View Complete document link (Component-A)"):
            # Now on Page-C, try to click Component-B to download PDF
            if wait_and_click(wait, By.ID, "pdfdoc", "PDF button (Component-B)"):
                return True
        return False
    except Exception as e:
        print(f"Error in Task B: {e}")
        return False

class WebDriverPool:
    def __init__(self, size):
        self.drivers = Queue(maxsize=size)
        for _ in range(size):
            driver = setup_driver()
            wait = WebDriverWait(driver, Config.WAIT_TIME)
            self.drivers.put((driver, wait))

    def get(self):
        return self.drivers.get()

    def release(self, driver_tuple):
        self.drivers.put(driver_tuple)

    def cleanup(self):
        while not self.drivers.empty():
            driver, _ = self.drivers.get()
            driver.quit()

def process_document_parallel(args):
    url, driver_pool, lock = args
    driver, wait = driver_pool.get()
    try:
        success = process_single_document(driver, wait, url, lock)
        return success
    finally:
        driver_pool.release((driver, wait))

def process_document_links(driver_pool, lock, current_url):
    try:
        driver, wait = driver_pool.get()
        driver.get(current_url)
        time.sleep(Config.PAGE_LOAD_DELAY)

        # Get all links inside Component-E
        links = wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "div.result_title a")))
        
        doc_urls = [link.get_attribute('href') for link in links]
        print(f"\nFound {len(doc_urls)} documents")
        driver_pool.release((driver, wait))

        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            args = [(url, driver_pool, lock) for url in doc_urls]
            results = executor.map(process_document_parallel, args)
        
        return True

    except Exception as e:
        print(f"Error processing document links: {str(e)}")
        return False

def click_next_page(driver, wait):
    try:
        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        # Find Next button (Component-D inside Component-C)
        next_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//div[@class='bottom']//a[text()='Next']")))
        
        if next_button:
            print("\nFound Next button")
            next_url = next_button.get_attribute('href')
            return next_url
        return None

    except Exception as e:
        print(f"Error clicking Next button: {str(e)}")
        return None

def main():
    os.makedirs(Config.PDF_DIR, exist_ok=True)
    driver_pool = WebDriverPool(Config.MAX_WORKERS)
    lock = Lock()
    
    try:
        current_search_url = Config.INITIAL_SEARCH_URL
        page_num = 1
        
        while True:
            print(f"\nProcessing page {page_num}")
            print(f"Current URL: {current_search_url}")
            
            if not process_document_links(driver_pool, lock, current_search_url):
                print("Failed to process documents on current page")
                break
            
            driver, wait = driver_pool.get()
            driver.get(current_search_url)
            time.sleep(Config.PAGE_LOAD_DELAY)
            new_url = click_next_page(driver, wait)
            driver_pool.release((driver, wait))
            
            if not new_url:
                print("No more pages to process")
                break
                
            current_search_url = new_url
            page_num += 1
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        print("\nClosing browsers...")
        driver_pool.cleanup()

if __name__ == "__main__":
    main()