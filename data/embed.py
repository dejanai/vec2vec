#!/usr/bin/env python3
"""
Unified script to generate both Gemini and MxbAI embeddings for dejan.ai content
Implements a staged approach:
- Stage 0: Check if content is already scraped and saved
- Stage 1: Scrape pages and save to content.csv if needed
- Stage 2: Generate embeddings using both Gemini and MxbAI models
"""

import os
import csv
import time
import json
import pandas as pd
import trafilatura
import numpy as np
from urllib.parse import urlparse
import requests
from requests.exceptions import RequestException
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# File paths - all relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URL_FILE = os.path.join(SCRIPT_DIR, "clean_url_list.txt")
CONTENT_CSV = os.path.join(SCRIPT_DIR, "content.csv")
GEMINI_CSV = os.path.join(SCRIPT_DIR, "gemini.csv")
MXBAI_CSV = os.path.join(SCRIPT_DIR, "mxbai.csv")

def check_for_existing_content():
    """Check if content.csv already exists and has data"""
    if os.path.exists(CONTENT_CSV):
        try:
            df = pd.read_csv(CONTENT_CSV)
            if len(df) > 0:
                logger.info(f"Found existing content file with {len(df)} entries")
                return True
        except Exception as e:
            logger.warning(f"Error reading existing content file: {e}")
    
    logger.info("No valid existing content file found")
    return False

def extract_text_from_url(url):
    """Extract main text content from a URL using trafilatura"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            logger.warning(f"Failed to download content from {url}")
            return None
        
        # Extract the main text content
        text = trafilatura.extract(downloaded, include_comments=False, 
                                  include_tables=False, no_fallback=False)
        
        if not text or len(text.strip()) < 50:  # Ensure we have meaningful content
            logger.warning(f"Insufficient text content extracted from {url}")
            return None
            
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return None

def load_urls_from_file():
    """Load URLs directly from clean_url_list.txt"""
    urls = []
    try:
        with open(URL_FILE, 'r') as f:
            for line in f:
                url = line.strip()
                if url:  # Skip empty lines
                    urls.append(url)
        logger.info(f"Loaded {len(urls)} URLs from {URL_FILE}")
        return urls
    except Exception as e:
        logger.error(f"Error loading URLs from {URL_FILE}: {e}")
        return []

def scrape_and_save_content(urls):
    """Scrape content from URLs and save to CSV"""
    results = []
    for i, url in enumerate(urls):
        logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
        
        # Extract text
        text = extract_text_from_url(url)
        if text is None:
            logger.warning(f"Skipping URL due to text extraction failure: {url}")
            continue
        
        # Add to results
        results.append({
            'url': url,
            'text': text
        })
        
        # Add a small delay to avoid overloading
        time.sleep(0.5)
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(CONTENT_CSV, index=False)
        logger.info(f"Successfully saved content for {len(results)} URLs to {CONTENT_CSV}")
        return df
    else:
        logger.error("No content was scraped. Check logs for errors.")
        return None

def load_content_from_csv():
    """Load content from CSV file"""
    try:
        df = pd.read_csv(CONTENT_CSV)
        logger.info(f"Loaded {len(df)} entries from {CONTENT_CSV}")
        return df
    except Exception as e:
        logger.error(f"Error loading content from CSV: {e}")
        return None

def generate_gemini_embeddings(df):
    """Generate Gemini embeddings for content"""
    # Import here to avoid errors if not needed
    try:
        from google import genai
    except ImportError:
        logger.error("Failed to import google-generativeai. Please install with: pip install google-generativeai")
        return None
    
    # Load API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    model = "gemini-embedding-exp-03-07"
    
    results = []
    for i, row in df.iterrows():
        url = row['url']
        text = row['text']
        
        logger.info(f"Generating Gemini embedding for URL {i+1}/{len(df)}: {url}")
        
        # Generate embedding with retry logic
        embedding = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=text
                )
                embedding = result.embeddings
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        
        if embedding is None:
            logger.warning(f"Failed to generate Gemini embedding for {url}")
            continue
        
        results.append({
            'url': url,
            'embedding': embedding
        })
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    # Save results to CSV
    if results:
        with open(GEMINI_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'embedding'])
            for result in results:
                writer.writerow([result['url'], ','.join(map(str, result['embedding']))])
        
        logger.info(f"Successfully saved Gemini embeddings for {len(results)} URLs to {GEMINI_CSV}")
        return results
    else:
        logger.error("No Gemini embeddings were generated. Check logs for errors.")
        return None

def generate_mxbai_embeddings(df):
    """Generate MxbAI embeddings for content"""
    # Import here to avoid errors if not needed
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("Failed to import sentence-transformers. Please install with: pip install sentence-transformers")
        return None
    
    # Load MxbAI model
    logger.info("Loading MxbAI Embed Large v1 model...")
    try:
        model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load MxbAI model: {e}")
        return None
    
    results = []
    for i, row in df.iterrows():
        url = row['url']
        text = row['text']
        
        logger.info(f"Generating MxbAI embedding for URL {i+1}/{len(df)}: {url}")
        
        # Generate embedding with retry logic
        embedding = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                embedding = model.encode(text)
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        
        if embedding is None:
            logger.warning(f"Failed to generate MxbAI embedding for {url}")
            continue
        
        results.append({
            'url': url,
            'embedding': embedding
        })
        
        # Add a small delay
        time.sleep(0.5)
    
    # Save results to CSV
    if results:
        with open(MXBAI_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'embedding'])
            for result in results:
                writer.writerow([result['url'], ','.join(map(str, result['embedding']))])
        
        logger.info(f"Successfully saved MxbAI embeddings for {len(results)} URLs to {MXBAI_CSV}")
        return results
    else:
        logger.error("No MxbAI embeddings were generated. Check logs for errors.")
        return None

def main():
    """Main function implementing the staged approach"""
    logger.info("Starting unified embedding generation process")
    
    # Stage 0: Check if content is already scraped and saved
    content_exists = check_for_existing_content()
    content_df = None
    
    if not content_exists:
        # Stage 1: Scrape pages and save to content.csv
        logger.info("Stage 1: Scraping content from URLs")
        urls = load_urls_from_file()
        if not urls:
            logger.error("No URLs found in clean_url_list.txt. Exiting.")
            return
        
        content_df = scrape_and_save_content(urls)
        if content_df is None:
            logger.error("Failed to scrape content. Exiting.")
            return
    else:
        # Load existing content
        content_df = load_content_from_csv()
        if content_df is None:
            logger.error("Failed to load existing content. Exiting.")
            return
    
    # Stage 2: Generate embeddings
    logger.info("Stage 2: Generating embeddings")
    
    # Generate Gemini embeddings
    logger.info("Generating Gemini embeddings")
    gemini_results = generate_gemini_embeddings(content_df)
    
    # Generate MxbAI embeddings
    logger.info("Generating MxbAI embeddings")
    mxbai_results = generate_mxbai_embeddings(content_df)
    
    # Report results
    logger.info("Embedding generation process completed")
    logger.info(f"Content saved to: {CONTENT_CSV}")
    if gemini_results:
        logger.info(f"Gemini embeddings saved to: {GEMINI_CSV}")
    if mxbai_results:
        logger.info(f"MxbAI embeddings saved to: {MXBAI_CSV}")

if __name__ == "__main__":
    main()
