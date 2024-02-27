## CLIP-as-a-Service

A Minimal Embedding Framework for Extracting Embeddings for Images and Sentences with CLIP. It simplifies the process of embedding extraction, providing ready-to-use functions for both batch and single instance embedding tasks. The framework eliminates the need for extensive technical expertise, making it accessible to a wider range of users.

## Getting Started

To embark on your journey with _CLIP-as-a-service_, follow these simple steps:

### Installation

Install _CLIP-as-a-service_ using the bellow installation instructions.

```bash
$ cd this-repo
$ DOCKER_BUILDKIT=1 docker build -t nhtlongcs/first-clip:latest services/clip-embedding/
$ #config and rename .env.dev to .env
$ docker-compose up -d # Start the _CLIP-as-a-service_ server to enable embedding extraction
```

### Documentation

Here are some illustrative examples of using _CLIP-as-a-service_:

<table>
<tr>
<td> via HTTP </td>
<td> via python </td>
</tr>
<tr>
<td>

```bash
curl \
-X POST http://localhost:{CLIP_PORT} \
-H 'Content-Type: application/json' \
-d
```

</td>
<td>

```python
# python example
```

</td>
</tr>
</table>

## Key Features

**Ready-to-Use Functions**: _CLIP-as-a-service_ eliminates the need for complex code or intricate model configurations. It provides ready-to-use functions for both batch and single instance embedding tasks, simplifying the process of embedding extraction.

**HTTPS Connection with FastAPI**: _CLIP-as-a-service_ supports HTTPS connections, enabling secure communication with your HTTP client of choice. This ensures the confidentiality and integrity of data during embedding extraction.

**Strong Community Support**: _CLIP-as-a-service_ benefits from a vibrant and supportive community of users and developers. This ensures prompt assistance and ongoing contributions to the framework's development.

**Easy Scalability with Docker**: _CLIP-as-a-service_ can be seamlessly scaled using Docker containers, making it suitable for handling large workloads efficiently. This allows for effortless deployment and management of embedding extraction tasks.

**GPU Support for Enhanced Performance**: _CLIP-as-a-service_ supports GPU acceleration, leveraging the computational power of GPUs to significantly improve embedding extraction performance. This is particularly beneficial for large-scale embedding tasks.

## Applications

_CLIP-as-a-service_ finds its application in various domains, including:

ReID (Person Re-identification): _CLIP-as-a-service_ can be used to extract embeddings for person images, enabling accurate re-identification of individuals across different camera views.

Image Retrieval: _CLIP-as-a-service_ facilitates image retrieval tasks by extracting embeddings that capture semantic similarities between images. This enables efficient retrieval of relevant images based on a query image.

## Conclusion

_CLIP-as-a-service_ stands out as a user-friendly embedding framework, providing a seamless experience for extracting embeddings from images and sentences. Its ready-to-use functions, HTTPS connectivity, strong community support, easy scalability, and GPU support make it an invaluable tool for a wide range of applications, including reID systems, image retrieval, image captioning, text summarization, and question answering.

## Acknowledgments

- [OpenAI CLIP]()
- Inspired by [Jina AI: Clip-as-Service]()
