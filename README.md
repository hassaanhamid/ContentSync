# ContentSync

An AI-powered web application that analyzes the semantic alignment between textual content and images. Built with Python, this tool utilizes OpenAI's CLIP (Contrastive Language-Image Pretraining) model to evaluate how cohesively an uploaded image matches a given article or text block.

**Live Demo:** Try the application directly in your browser at [https://contentsync.streamlit.app/](https://contentsync.streamlit.app/).

## Tech Stack

* **Language:** Python
* **Framework:** Streamlit
* **Machine Learning:** PyTorch, OpenAI CLIP

## Features

* **Semantic Alignment Scoring:** Leverages CLIP's multimodal embeddings to calculate a similarity score between text and visual inputs.
* **Interactive Web Interface:** A streamlined UI built with Streamlit allows users to easily paste text, upload an image, and instantly receive evaluation metrics.
* **Real-Time Analysis:** Processes inputs dynamically through the ML pipeline to provide immediate feedback on visual and textual alignment.

## How to Use the Web App

1. Visit the live application at (https://contentsync.streamlit.app/).
2. Paste your article or text block into the provided input field.
3. Upload the image you want to evaluate.
4. Click the analyze button to generate the semantic alignment score.
