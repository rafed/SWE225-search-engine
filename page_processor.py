import os
import sys
import json
from bs4 import BeautifulSoup
from text_processor import tokenize, stem_words
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse, parse_qsl, urlencode, quote, unquote
from simhashdb import SimhashManager
from pathlib import Path
import tqdm

from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin, unquote
import re
import ipaddress

simhash = SimhashManager()

def get_file_content(filepath):
    encoding_type = ''
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'encoding' in data:
                encoding_type = data['encoding']
            if 'url' in data:
                url = normalize_url(data['url'])
            if 'content' in data:
                content = data['content']
                if encoding_type:
                    content = content.encode(encoding_type).decode(encoding_type)
    except (json.JSONDecodeError, IOError) as e:
        print(f'Error in {filepath}: {e}')

    return url, content


def process_files(input_directory, output_directory):
    files = list(Path(input_directory).rglob('*.json'))

    for filepath in tqdm.tqdm(files):
        # print(filepath)
        url, content = get_file_content(filepath)
        if simhash.exists_duplicate(url, content):
            continue

        content_dict = {}
        content_dict['url'] = url

        soup = BeautifulSoup(content, "lxml")
        text_category = categorize_text(url, soup)
        
        for category, texts in text_category.items():
            if category == 'anchor': 
                content_dict[category] = texts
                continue
            all_tokens = []
            for text in texts:
                tokens = tokenize(text)
                stemmed_tokens = stem_words(tokens)
                all_tokens.extend(stemmed_tokens)
            
            content_dict[category] = all_tokens
        
        output_filename = filepath.stem + '_processed' + filepath.suffix
        save_file(output_filename, output_directory, content_dict)


def save_file(filename, output_directory, content_dict):          
    output_filepath = os.path.join(output_directory, filename)

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(content_dict, f, ensure_ascii=False, indent=4)


def is_valid_ipv4(hostname):
    """Check if the hostname is a valid IPv4 address."""
    try:
        # Remove brackets if present (for IPv6-style notation)
        hostname = hostname.strip('[]')
        return ipaddress.ip_address(hostname).version == 4
    except ValueError:
        return False

def is_valid_domain(hostname):
    """Check if the hostname is a valid domain (e.g., example.com)."""
    # Updated regex to better handle domains and allow for more valid cases
    pattern = r"^(?!-)[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,}$"
    return bool(re.match(pattern, hostname)) or hostname == 'localhost'

def normalize_url(url):
    try:
        url = url.strip()
        if not url:
            return None

        if url.startswith(("mailto:", "javascript:", "data:")):
            return None  

        if not url.startswith(('http://', 'https://', 'ftp://')):
            url = "http://" + url

        parsed = urlparse(url)
        hostname = parsed.hostname.strip("[]") if parsed.hostname else None
        
        if not hostname:
            print(f"Skipping URL without hostname: {url}")
            return None

        if not (is_valid_domain(hostname) or is_valid_ipv4(hostname)):
            print(f"Skipping URL with invalid hostname: {url}")
            return None

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        if (scheme == "http" and netloc.endswith(":80")) or (scheme == "https" and netloc.endswith(":443")):
            netloc = netloc.rsplit(":", 1)[0]

        path = unquote(parsed.path).rstrip("/") if parsed.path and parsed.path != "/" else "/"
        query = urlencode(sorted(parse_qsl(parsed.query))) if parsed.query else ""
        fragment = ""

        normalized = urlunparse((scheme, netloc, path, "", query, fragment))
        return normalized

    except Exception as e:
        print(f"Skipping invalid URL '{url}': {e}")
        return None

def remove_tags_and_content(soup):
    tags_to_remove = ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'strong']

    for tag in tags_to_remove:
        for element in soup.find_all(tag):
            if element.parent:
                element.insert_before(' ')
                element.decompose()

    return soup


def categorize_text(url, soup):
    anchors = []
    for a in soup.find_all('a', href=True):
        try:
            link_url = urljoin(url, a['href'])
            normalized_url = normalize_url(link_url)
            if normalized_url:
                anchors.append(normalized_url)
        except Exception as e:
            print(f"Skipping invalid URL '{url} + {a['href']}': {e}")

    text_category = {
        'title': [soup.title.string] if soup.title else [],
        'h1': [h1.get_text(strip=True) for h1 in soup.find_all('h1')],
        'h2': [h2.get_text(strip=True) for h2 in soup.find_all('h2')],
        'h3': [h3.get_text(strip=True) for h3 in soup.find_all('h3')],
        'h4': [h3.get_text(strip=True) for h3 in soup.find_all('h4')],
        'h5': [h3.get_text(strip=True) for h3 in soup.find_all('h5')],
        'h6': [h3.get_text(strip=True) for h3 in soup.find_all('h6')],
        'bold': [b.get_text(strip=True) for b in soup.find_all(['b', 'strong'])],
        'anchor': anchors
    }
    soup = remove_tags_and_content(soup)
    text_category['other_text'] = [' '.join(soup.get_text().split())]
    
    return text_category


if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = './processed_files'

    os.makedirs(output_directory, exist_ok=True)

    process_files(input_directory, output_directory)