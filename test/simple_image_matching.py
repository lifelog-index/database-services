import os
import asyncio
from dotenv import load_dotenv
from src.image_analyzer import ImageAnalyzer
from src.similarity_search import ImageSimilaritySearch


BASE_PATH = os.path.abspath("../data")


async def main():
    """Takes images form the data directory, analyzes and adds them to elastic, then tries to look them up again"""
    load_dotenv()
    search = ImageSimilaritySearch(os.environ["ELASTIC_HOST"], os.environ["ELASTIC_USER"],
                                   os.environ["ELASTIC_PASSWORD"], use_temp_ix=True)
    await search.ensure_indexes_exists()
    ex = ImageAnalyzer()

    file_names = [x for x in os.listdir(BASE_PATH) if os.path.isfile(os.path.join(BASE_PATH, x))][:3]
    print(file_names)

    feats = [
        (ex.extract_descriptors(os.path.join(BASE_PATH, x)), 
         ex.extract_exif(os.path.join(BASE_PATH, x)),
         ex.extract_noiseprint(os.path.join(BASE_PATH, x)))
        for x in file_names
    ]

    for name, (desc, exif, noiseprint) in zip(file_names, feats):
        if await search.has_image(name):
            print(f"Image at {name} already stored in index, continuing...")
            continue
        await search.add_image(desc, "http://example.com/", name, exif, noiseprint)

    for name, (desc, _, _) in zip(file_names, feats):
        img_doc, score = await search.search_image(desc)
        print(name, f"Success = {img_doc['img_url'] == name}")


if __name__ == "__main__":
    asyncio.run(main())
