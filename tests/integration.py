import os
import shutil
import asyncio
from dotenv import load_dotenv
from src.web_scraper import WebScraper
from src.factiverse import FactiverseApi
from src.image_analyzer import ImageAnalyzer
from src.similarity_search import ImageSimilaritySearch


load_dotenv()


async def main():
    endpoint = os.environ["AUTH0_ENDPOINT"]
    client_id = os.environ["AUTH0_CLIENT_ID"]
    client_secret = os.environ["AUTH0_CLIENT_SECRET"]

    factiverse = FactiverseApi(endpoint, client_id, client_secret)
    res = await factiverse.claim_search("climate change", languages="en", size=10)

    article_urls = [x.url for x in res.search_results]

    print(f"Got {len(article_urls)} articles from Factiverse")

    scraper = WebScraper(article_urls)
    await scraper.fetch_img_sources()
    await scraper.fetch_img_contents()

    print(f"Got {len(scraper.image_links_files.keys())} images from articles")

    ia = ImageAnalyzer()
    feats = [
        (ia.extract_descriptors(v), ia.extract_exif(v), ia.extract_noiseprint(v))
        for _, v in scraper.image_links_files.items()
    ]

    print(f"Analyzed {len(feats)} images")

    search = ImageSimilaritySearch(os.environ["ELASTIC_HOST"], os.environ["ELASTIC_USER"],
                                   os.environ["ELASTIC_PASSWORD"])
    await search.ensure_indexes_exists()

    ct = 0
    for (url, _), (desc, exif, noiseprint) in zip(scraper.image_links_files.items(), feats):
        if desc is None:
            print(f"No descriptors discovered in {url}, continuing...")
            continue
        if await search.has_image(url):
            print(f"Image at {url} already stored in index, continuing...")
            continue

        article = [k for k, v in scraper.articles_images.items() if url in v][0]
        await search.add_image(desc, article, url, exif, noiseprint)
        ct += 1

    print(f"Added {ct} images to the index")

    if not os.path.isdir("../data/fetched"):
        os.mkdir("../data/fetched")

    for url, file in scraper.image_links_files.items():
        name = file.split('/')[-1]
        shutil.copy(file, f"../data/fetched/{name}")


if __name__ == "__main__":
    asyncio.run(main())
