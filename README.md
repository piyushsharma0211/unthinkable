
# 🛍️ Visual Product Matcher

This project is a visual product search engine that allows users to upload an image or paste an image URL to find visually similar products from a preloaded database. It uses OpenAI's CLIP Vision Transformer (ViT-B-32) model for feature extraction and cosine similarity for matching images.

---

## Demo Video

[Watch the demo video here](https://github.com/user-attachments/assets/5a613ee7-8b08-4546-9ffe-5f65f10113ed)

---

## Features

- **Image Upload & URL Input**: Upload images directly or paste image URLs.
- **Feature Extraction**: Uses the CLIP ViT-B-32 model to extract rich image embeddings.
- **Similarity Search**: Finds and ranks similar product images from the database using cosine similarity.
- **Adjustable Parameters**: Customize the number of similar products to display.
- **Performance Optimizations**: Caching with Streamlit for faster subsequent queries.

---

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/nasirovsh/ecommerce-visual-search.git
    cd ecommerce-visual-search
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate    # Windows: `.venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. *(Optional)* For better file-watching performance, install watchdog:
    - macOS:
      ```bash
      xcode-select --install
      pip install watchdog
      ```
    - Windows:
      ```bash
      pip install watchdog
      ```

---

## Usage

1. **Prepare your product image database:**
   - Place product images (JPG/PNG) in the `db/` directory.
   - Make sure images are clear and representative of your products.
   - The demo uses 60+ product images.

2. **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```

3. **Interact with the app:**
   - Upload a product image or paste its URL.
   - Choose how many similar products to display.
   - Click **Find Similar Products** to view results.

---

## Project Structure

````

ecommerce-visual-search/
├── main.py                # Main Streamlit app script
├── db/                    # Folder containing product images
├── requirements.txt       # Python dependencies
├── README.md              # This documentation file
└── LICENSE                # Project license

```

---

## Dependencies

- `streamlit` — Web app framework
- `torch` — PyTorch for model inference
- `sentence-transformers` — For CLIP model and embedding extraction
- `pillow` — Image processing
- `numpy` — Numerical operations
- `scikit-learn` — For cosine similarity calculations
- `certifi` — SSL certificates

*(See `requirements.txt` for exact versions.)*

---

## How It Works

1. On startup, the CLIP ViT-B-32 model loads and precomputes embeddings for all database images.
2. When a user uploads an image or submits a URL, the app extracts its feature embedding.
3. Cosine similarity scores are computed between the query embedding and all database embeddings.
4. The app displays the top-N visually similar products based on these scores.

---

## Troubleshooting

- Ensure your images are in supported formats (JPG/PNG) and not corrupted.
- If the app runs slowly initially, wait for caching to complete; subsequent runs will be faster.
- If memory issues occur, reduce the number of database images or increase available RAM.

---

## Acknowledgements

- OpenAI's CLIP model for powerful image embeddings.
- Streamlit for an amazing app development framework.
- Thanks to the community for open-source resources and inspiration.

---

## License

MIT License © Your Name

---

## Contact

Questions or feedback? Reach out via email at your.email@example.com or connect on [LinkedIn](https://linkedin.com/in/yourprofile).

---

*Built with ❤️ using Streamlit and OpenAI CLIP*
```

---

