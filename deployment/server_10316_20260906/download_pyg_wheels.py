from pathlib import Path
from html.parser import HTMLParser
from urllib.request import urlopen
from urllib.parse import urljoin, unquote, urlparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time

INDEX = 'https://data.pyg.org/whl/torch-2.10.0+cu128.html'
OUT = Path(__file__).resolve().parent / 'wheelhouse'
OUT.mkdir(exist_ok=True)

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href: self.links.append(urljoin(INDEX, href))

parser = Links()
with urlopen(INDEX, timeout=60) as response:
    parser.feed(response.read().decode())
wanted = ['pyg_lib-0.8.0+pt210cu128-', 'torch_scatter-2.1.2+pt210cu128-',
          'torch_sparse-0.6.18+pt210cu128-', 'torch_cluster-1.6.3+pt210cu128-']
urls = []
for prefix in wanted:
    choices = [url for url in parser.links if unquote(urlparse(url).path.rsplit('/',1)[-1]).startswith(prefix)
               and 'cp310' in url and 'x86_64' in url and 'win_' not in url]
    assert len(choices) == 1, (prefix, choices)
    urls.append(choices[0])

def download(url):
    name = unquote(urlparse(url).path.rsplit('/',1)[-1])
    path = OUT / name
    started = time.time()
    with urlopen(url, timeout=90) as response, path.with_suffix('.part').open('wb') as target:
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            target.write(chunk)
    path.with_suffix('.part').replace(path)
    result = {'file':name, 'url':url, 'bytes':path.stat().st_size,
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'seconds':round(time.time()-started,2)}
    print(json.dumps(result), flush=True)
    return result

with ThreadPoolExecutor(4) as pool:
    rows = list(pool.map(download, urls))
(OUT / 'wheel_manifest.json').write_text(json.dumps({'index':INDEX,'wheels':rows},indent=2)+'\n')
print(json.dumps({'completed':len(rows),'bytes':sum(row['bytes'] for row in rows)}))
