import os
import sys
import json
from bs4 import BeautifulSoup
from text_processor import tokenize, stem_words
from urllib.parse import urldefrag, urljoin
from simhashdb import SimhashManager
from pathlib import Path
import tqdm

simhash = SimhashManager()

def get_file_content(filepath):
    encoding_type = ''
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'encoding' in data:
                encoding_type = data['encoding']
            if 'url' in data:
                url, _ = urldefrag(data['url'])
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
                    

def remove_tags_and_content(soup):
    tags_to_remove = ['title', 'h1', 'h2', 'h3', 'b', 'strong']

    for tag in tags_to_remove:
        for element in soup.find_all(tag):
            if element.parent:
                element.insert_before(' ')
                element.decompose()

    return soup


def categorize_text(url, soup):
    text_category = {
        'title': [soup.title.string] if soup.title else [],
        'h1': [h1.get_text(strip=True) for h1 in soup.find_all('h1')],
        'h2': [h2.get_text(strip=True) for h2 in soup.find_all('h2')],
        'h3': [h3.get_text(strip=True) for h3 in soup.find_all('h3')],
        'bold': [b.get_text(strip=True) for b in soup.find_all(['b', 'strong'])],
        'anchor': [urldefrag(urljoin(url, a['href']))[0] for a in soup.find_all('a', href=True)] # write code to process URLs http://fr.ics.uci.edu/haiti/ vs http://fr.ics.uci.edu/haiti
    }
    
    soup = remove_tags_and_content(soup)
    text_category['other_text'] = [' '.join(soup.get_text().split())]
    
    return text_category


if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = './processed_files'

    os.makedirs(output_directory, exist_ok=True)

    process_files(input_directory, output_directory)
        