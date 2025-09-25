"""
This module handles the extraction of text from English and Japanese Wikipedia articles.
It uses the 'wikipedia-api' library for easy access to article content.
"""
import wikipediaapi
from config import Config
import logging
import json

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_wiki_articles(en_title: str, ja_title: str, num_paragraphs: int = 10):
    """
    Fetches text from a specified number of paragraphs of English and Japanese Wikipedia articles.

    Args:
        en_title (str): The title of the English Wikipedia article.
        ja_title (str): The title of the Japanese Wikipedia article.
        num_paragraphs (int): The number of paragraphs to extract from each article.

    Returns:
        dict: A dictionary containing the extracted texts and their languages,
              or None if fetching fails.
    """
    try:
        wiki_en = wikipediaapi.Wikipedia(language='en', user_agent='MechanisticInterpretability/v1.0 (sharanya.ranka@gmail.com)')
        wiki_ja = wikipediaapi.Wikipedia(language='ja', user_agent='MechanisticInterpretability/v1.0 (sharanya.ranka@gmail.com)')

        en_page = wiki_en.page(en_title)
        ja_page = wiki_ja.page(ja_title)

        if not en_page.exists() or not ja_page.exists():
            logger.error("One or both Wikipedia articles do not exist.")
            return None

        en_text = '\n'.join(en_page.text.split('\n')[2:2 + num_paragraphs]).strip()
        ja_text = '\n'.join(ja_page.text.split('\n')[2:2 + num_paragraphs]).strip()
        
        if not en_text or not ja_text:
            logger.error("Failed to extract text from articles. They may be too short.")
            return None
            
        return {
            "english": en_text,
            "japanese": ja_text
        }

    except Exception as e:
        logger.error(f"An error occurred while fetching Wikipedia articles: {e}")
        return None


if __name__ == "__main__":
    d = fetch_wiki_articles(Config.EN_ARTICLE_TITLE, Config.JA_ARTICLE_TITLE, 2)
    print(json.dumps(d, indent=4))
