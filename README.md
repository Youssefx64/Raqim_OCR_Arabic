# Raqim_OCR_Arabic (رقيم)

Raqim is an Arabic handwritten-text recognition (OCR/HTR) model. It fine-tunes
[`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
with LoRA on [`aamijar/muharaf-public`](https://huggingface.co/datasets/aamijar/muharaf-public),
a dataset of historical handwritten Arabic manuscript lines.

## Model

The fine-tuned, merged model and its LoRA adapter are hosted on Hugging Face:
[`Youssefx64/Raqim_OCR_Arabic`](https://huggingface.co/Youssefx64/Raqim_OCR_Arabic).
The notebook creates this repository as **private**, so you may need access
permission and Hugging Face authentication to download it.

- Base model: `Qwen/Qwen3-VL-2B-Instruct` (~2B parameters)
- Fine-tuning: LoRA on the language model and vision tower; 2 epochs
- Input: an image crop of one handwritten text line
- Output: the Arabic transcription. Full-page images need line segmentation first.
- Peak training VRAM: 10.9 GB

## Results

| Test split | CER | WER |
|---|---:|---:|
| Base model (zero-shot, n=300) | 1.1281 | 1.1144 |
| Fine-tuned model (n=1,334) | 0.1737 | 0.4494 |

## Usage

After getting access to the private model repository, authenticate with
Hugging Face and load the model and processor:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

repo = "Youssefx64/Raqim_OCR_Arabic"
processor = AutoProcessor.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(
    repo, dtype="auto", device_map="auto"
)

prompt = processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text":
         "Transcribe the Arabic handwriting in this image exactly as written."}
    ]}],
    add_generation_prompt=True,
    tokenize=False,
)
image = Image.open("line.png").convert("RGB")
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False)
transcription = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
)[0]
print(transcription)
```

## Project files

- [Training notebook](./Raqim_OCR_Arabic_Handwriting_Finetune.ipynb): data preparation, training, evaluation, model merge, and Hugging Face upload.
- [Metrics](./reports/metrics.json)
- [Per-sample predictions](./reports/test_predictions.csv)

## License

The Muharaf dataset is licensed under CC BY-NC-SA 4.0. This model is
non-commercial and share-alike under the same terms. See [LICENSE.md](./LICENSE.md).
Please cite *Muharaf: Manuscripts of Handwritten Arabic Dataset for Cursive
Text Recognition* ([arXiv:2406.09630](https://arxiv.org/abs/2406.09630)).
