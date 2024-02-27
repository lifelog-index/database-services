import os
import cv2
import time
import asyncio
import numpy as np
from dotenv import load_dotenv
from src.image_analyzer import ImageAnalyzer
from src.similarity_search import ImageSimilaritySearch


# Dataset downloaded from http://images.cocodataset.org/zips/val2017.zip
BASE_PATH = os.path.abspath("../data/coco-val2017")


async def main():
    """Takes images form the data directory, analyzes and adds them to elastic, then tries to look them up again"""
    load_dotenv()
    search = ImageSimilaritySearch(os.environ["ELASTIC_HOST"], os.environ["ELASTIC_USER"],
                                   os.environ["ELASTIC_PASSWORD"], use_temp_ix=True, keep_temp_ix=True)
    await search.ensure_indexes_exists()
    print(f"Using image index: {search.image_ix_name}, vector index: {search.vector_ix_name}")
    ia = ImageAnalyzer()

    img_ct = 10
    file_names = os.listdir(BASE_PATH)[:img_ct]
    print(file_names)

    _ = await import_images(ia, file_names, search)  # Insertion takes about 3 seconds per image, 1442 feats per

    rotated, cropped, noised, compressed = get_modified_images(file_names, ia)

    print("Searching with modified images")
    for name, r, cr, n, co in zip(file_names, rotated, cropped, noised, compressed):
        start = time.perf_counter()
        r_doc, r_score = await search.search_image(r)
        cr_doc, cr_score = await search.search_image(cr)
        n_doc, n_score = await search.search_image(n)
        co_doc, co_score = await search.search_image(co)

        print(name, f"rotate = {r_doc['img_url'] == name}, crop = {cr_doc['img_url'] == name}, "
                    f"noise = {n_doc['img_url'] == name}, compress = {co_doc['img_url'] == name}, "
                    f"time = {time.perf_counter() - start:.0f}s")


async def import_images(ia: ImageAnalyzer, file_names: list[str], search: ImageSimilaritySearch) -> list[np.ndarray]:
    feats = [
        (ia.extract_descriptors(os.path.join(BASE_PATH, x)),
         ia.extract_exif(os.path.join(BASE_PATH, x)),
         ia.extract_noiseprint(os.path.join(BASE_PATH, x)))
        for x in file_names
    ]

    print(f"Extracted features for {len(file_names)} images")

    start = time.perf_counter()
    for name, (desc, exif, noiseprint) in zip(file_names, feats):
        await search.add_image(desc, "", name, exif, noiseprint)

    print(f"Added {len(file_names)} images to the db in {(time.perf_counter() - start):.2f}s")
    return [x[0] for x in feats]


def get_modified_images(file_names: list[str], ia: ImageAnalyzer):
    rotated = []
    cropped = []
    noised = []
    compressed = []
    
    for name in file_names:
        im = cv2.imread(os.path.join(BASE_PATH, name))

        # Rotate by 50 deg
        image_center = tuple(np.array(im.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, 50, 1.0)
        aff2 = cv2.warpAffine(im, rot_mat, (im.shape[1::-1]))

        p = os.path.join(BASE_PATH, "..", "coco-val2017-rotated")
        if not os.path.isdir(p):
            os.mkdir(p)
        p = os.path.join(p, name)

        cv2.imwrite(p, aff2)
        rotated.append(ia.extract_descriptors(p))

        # Crop away outer 30%
        w_ex = 15 * (im.shape[0] // 100)
        h_ex = 15 * (im.shape[1] // 100)
        cropped_im = im[w_ex:im.shape[0]-w_ex, h_ex:im.shape[1]-h_ex]

        p = os.path.join(BASE_PATH, "..", "coco-val2017-cropped")
        if not os.path.isdir(p):
            os.mkdir(p)
        p = os.path.join(p, name)

        cv2.imwrite(p, cropped_im)
        cropped.append(ia.extract_descriptors(p))

        # Add gaussian noise
        gauss_noise = np.zeros(im.shape, dtype=np.uint8)
        cv2.randn(gauss_noise, 128, 20)
        gauss_noise = (gauss_noise * 0.5).astype(np.uint8)
        noised_im = cv2.add(im, gauss_noise)

        p = os.path.join(BASE_PATH, "..", "coco-val2017-noised")
        if not os.path.isdir(p):
            os.mkdir(p)
        p = os.path.join(p, name)

        cv2.imwrite(p, noised_im)
        noised.append(ia.extract_descriptors(p))

        # Add jpeg compression
        p = os.path.join(BASE_PATH, "..", "coco-val2017-jpeg-compressed")
        if not os.path.isdir(p):
            os.mkdir(p)
        p = os.path.join(p, name)

        cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Default is 95, lower is more compressed.
        compressed.append(ia.extract_descriptors(p))

    return rotated, cropped, noised, compressed


if __name__ == "__main__":
    asyncio.run(main())
