# MemoryCard Project Page

A responsive, dependency-free static project page for:

**MemoryCard: Topic-Aware Multi-Modal Clue Compression for Long-Video Question Answering**

## Structure

```text
MemoryCard-project-page/
├── index.html
├── css/styles.css
├── js/main.js
└── assets/
    ├── figure1-motivation.png
    ├── figure2-framework.png
    ├── figure3-category-wise.png
    ├── figure4-resolution.png
    ├── figure5-selection.png
    ├── figure6-selfread.png
    ├── figure7-retrieval.png
    ├── favicon.svg
    └── memorycard-paper.pdf
```

## Preview locally

Open `index.html` directly, or run a local server:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Deploy with GitHub Pages

This project is published from the `gh-pages` branch of
[`NEUIR/MemoryCard`](https://github.com/NEUIR/MemoryCard).

In **Settings → Pages**, select **Deploy from a branch**, choose `gh-pages`
and `/ (root)`, then save. The site URL is:

`https://neuir.github.io/MemoryCard/`

## Customize

- Edit paper text and author links in `index.html`.
- Change colors in the `:root` block of `css/styles.css`.
- Update benchmark values in `js/main.js` and the static table in `index.html`.
- Replace the BibTeX entry when the paper receives a proceedings citation.

The page uses no build tool and no third-party JavaScript library. Paper figures and the PDF remain subject to the paper authors' rights; the surrounding webpage code may be adapted for the project.
## Design notes

The hero uses a two-line desktop title and a custom Memory Card bank illustration, so the paper motivation figure appears only once in the Motivation section.
