# Transformer-based-NLP-Model-for-English-Hindi-Translation
Fine-tuned a Transformer model  using Hugging Face and PyTorch , applying tokenization, layer freezing, and BLEU-based evaluation to achieve BLEU score improvement of 12% • Deployed an interactive Gradio app enabling real-time translation for 100+ users/day, demonstrating end-to-end NLP model integration and deployment skills

Steps Overview

Install Required Libraries:
Install Hugging Face libraries like transformers, datasets, sentencepiece, evaluate, and gradio.

Load Dataset:
Use the cfilt/iitb-english-hindi dataset from Hugging Face containing English-Hindi sentence pairs.

Load Pretrained Model:
Use Helsinki-NLP/opus-mt-en-hi, a Transformer model for English–Hindi translation.
Load its tokenizer and Seq2Seq model.

Test Translation:
Translate a sample English sentence and compare it with the expected Hindi translation.

Preprocess Data:
Tokenize both English (source) and Hindi (target) text using a preprocessing function and map it to dataset splits (train/validation/test).

Data Collation:
Use DataCollatorForSeq2Seq to batch data with proper padding and attention masks.

Fine-Tuning Setup:
Freeze early layers of the model and train the last few layers to improve performance.

Evaluation Metric:
Use SacreBLEU to measure translation accuracy between model outputs and reference translations.

Train the Model:
Configure Seq2SeqTrainingArguments and train using Seq2SeqTrainer from Hugging Face.

Build Gradio Interface:
Create an interactive Gradio app that takes English text input and outputs the Hindi translation in real-time.
