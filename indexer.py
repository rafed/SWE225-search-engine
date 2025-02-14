import os
import sys
import json
from bs4 import BeautifulSoup
from text_processor import tokenize, stem_words
from urllib.parse import urldefrag

def extract_urls_and_contents(directory):
    print('extracting urls and contents')
    urls = []
    contents = {}
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                encoding_type = ''
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'encoding' in data:
                            encoding_type = data['encoding']
                        if 'url' in data:
                            url, _ = urldefrag(data['url'])
                            urls.append(url)
                        if 'content' in data:
                            content = data['content']
                            if encoding_type:
                                content = content.encode(encoding_type).decode(encoding_type)                                
                            contents[filepath] = BeautifulSoup(content, "lxml")
                except (json.JSONDecodeError, IOError) as e:
                    print(f'Error in {filepath}: {e}')
    
    return urls, contents

def remove_tags_and_content(soup):
    tags_to_remove = ['title', 'h1', 'h2', 'h3', 'b', 'strong']

    for tag in tags_to_remove:
        for element in soup.find_all(tag):
            element.insert_before(' ')
            element.decompose()

    return soup


def categorize_text(soup):
    text_category = {
        'title': [soup.title.string] if soup.title else [],
        'h1': [h1.get_text(strip=True) for h1 in soup.find_all('h1')],
        'h2': [h2.get_text(strip=True) for h2 in soup.find_all('h2')],
        'h3': [h3.get_text(strip=True) for h3 in soup.find_all('h3')],
        'bold': [b.get_text(strip=True) for b in soup.find_all(['b', 'strong'])],
    }
    
    soup = remove_tags_and_content(soup)
    text_category['other_text'] = [' '.join(soup.get_text().split())]
    
    return text_category


if __name__ == '__main__':
    input_directory = sys.argv[1]
    output_directory = './processed_files'

    os.makedirs(output_directory, exist_ok=True)

    urls, contents = extract_urls_and_contents(input_directory)
    
    for url, content in zip(urls, contents.items()):
        print('processing', url)
        content_dict = {}
        content_dict['url'] = url

        filename = content[0]
        soup = content[1]
        text_category = categorize_text(soup)
        
        for category, texts in text_category.items():
            all_tokens = []
            for text in texts:
                tokens = tokenize(text)
                stemmed_tokens = stem_words(tokens)
                all_tokens.extend(stemmed_tokens)
            
            content_dict[category] = all_tokens
        
        
        filename = os.path.basename(filename).replace('.json', '_processed.json')
        output_filepath = os.path.join(output_directory, filename)

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(content_dict, f, ensure_ascii=False, indent=4)