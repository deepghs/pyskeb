from typing import List

import xurls


def extract_urls(text) -> List[str]:
    extractors = [
        xurls.StrictScheme('https://'),
        xurls.StrictScheme('http://'),
    ]
    urls = []
    for extractor in extractors:
        urls.extend(extractor.findall(text))

    return sorted(set(urls))
