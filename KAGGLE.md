# Run SoulX-Singer SVC on Kaggle

Kaggle is suitable for testing and using this app with a GPU, but it is not permanent hosting. Kaggle sessions stop when the session ends, and Gradio share links are temporary.

## Setup

1. Import `kaggle_soulx_singer.ipynb` into Kaggle.
2. Open **Notebook settings**.
3. Set **Accelerator** to **GPU** and enable **Internet**.
4. Run the cells in order.
5. Wait for the model downloads and copy the `gradio.live` URL printed by the final cell.
6. Keep the Kaggle session running while using the URL.

The notebook clones the patched `saintus-create/singer` repository, installs its dependencies, fixes the NumPy/PyArrow compatibility issue, verifies CUDA, and starts the SVC-only app with `GRADIO_SHARE=1`.

## First test

Use a 10–30 second target first. Start with:

- Diffusion steps: **32**
- CFG scale: **1.0**
- Target vocal separation: **Off** for a pure vocal target
- Target vocal separation: **On** only when the target includes accompaniment and the GPU can handle the extra preprocessing
- Auto pitch shift: **On**
- Auto mix accompaniment: **On** when you want the instrumental retained

After a successful short test, try 64 steps and longer audio.

## Important limitation

The Kaggle URL is a temporary Gradio share link. Gradio documents that share links proxy the running notebook and expire; they are not permanent hosting. For a permanent URL, deploy the verified app to a Hugging Face Space or another GPU host after the Kaggle test succeeds.

References:

- https://www.kaggle.com/docs/notebooks
- https://www.gradio.app/guides/sharing-your-app
- https://www.gradio.app/guides/understanding-gradio-share-links
