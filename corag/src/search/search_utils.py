import requests

from typing import List, Dict

from logger_config import logger

# TODO: Add node host logic
def search_by_http(query: str, host: str = '10.197.17.38', port: int = 8090) -> List[Dict]:
    url = f"http://{host}:{port}"
    response = requests.post(url, json={'query': query})

    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to get a response. Status code: {response.status_code}")
        return []
