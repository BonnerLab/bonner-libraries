import json
import requests

FIGSHARE_API_BASE_URL = "https://api.figshare.com/v2"


def get_url_dict(article_id: int) -> dict[str, str]:
    files = json.loads(
        requests.get(f"{FIGSHARE_API_BASE_URL}/articles/{article_id}/files").content
    )
    urls = {file["name"]: file["download_url"] for file in files}
    return urls
