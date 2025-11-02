# Getting Started with Large Language Models (LLMs)
## Introduction to LLMs
- Tasks
    - Summarizing
    - Generating
    - Translating
    - QnA
- Based on deep learning architectures
- Most commonly transformers
- Huge neural networks with lots of parameters and text data
### Using Hugging Face models
```python
from transformers import pipeline

summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")

text = "insert text here"

summary = summarizer(text, max_length=50)

print(output[0]["summary_text"])
```
- `clean_up_tokenization_spaces=True`

## Using pre-trained LLMs
### Text generation
```python
generator = pipeline(task="text-generation", model="distilgpt2")

prompt=""

output = generator(prompt, max_length=100, pad_token_id=generator.tokenizer.eos_token_id) # eod: end of sequence

print(output[0]["generated_text"])
```
- `pad_token_id`: fills in extra space up to max_length
- padding: adding tokens
- marks the end of meaningful text, learned through training
- `truncation = True`

### Language Translation
```python
translator = pipeline(task="translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")

text = ""
output = translator(text, clean_up_tokenization_spaces=True)

print(output[0]["translation_text"])
```
## Understanding the transformer
- Transformers
    - Deep learning architectures
    - processing, understanding, and generating text
    - used in most LLMs
    - Handle long text sequences in parallel
    - Architectures
        - Encoder-only
        - Decoder-only
        - Encoder-Decoder
### Encoder-only
- Understanding only
- no sequential output
- Tasks: 
    - text classification, 
    - sentiment analysis
    - extractive question-answering
- BERT based model
```python
llm = pipeline(model="bert-base-uncased")
llm.model
llm.model.config

llm.model.config.is_decoder
llm.model.config.is_encoder_decoder
```

### Decoder-only
- focus shifts to output
- Task
    - text generation
    - generative question-answering
- gpt models
```python
llm = pipeline(model="gpt2")
print(llm.model.config)
```

### Encoder-decoder
- understand and process the input and output
- Common tasks:
    - Translation
    - Summarization
- T5, BART models
```python
llm = pipeline(model="Helsinki-NLP/opus-mt-es-en")
```

# Fine-tuning LLMs
## Preparing for fine-tuning
- Pipelines `pipeline()`
    - Streamline tasks
    - Automatic model and tokenizer selection
    - Limited control
- Auto classes `AutoModel`
    - Customization
    - Manual adjustments
    - supports fine-tuning
### LLM lifecycle
- Pre-training
    - Broad data
    - learn general patterns
- Fine-tuning
    - domain specific
    - specialized tasks
```python
# loading a dataset for fine-tuning
from datasets import load_dataset

train_data = load_dataset("imdb", split="train")
train_data = data.shard(num_shards=4, index=0)

test_data = load_dataset("imdb", split="test")
test_data = data.shard(num_shards=4, index=0)

# Auto classes
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the data
tokenized_training_data = tokenizer(train_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64)

tokenized_test_data = tokenizer(test_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64)
```
- Tokenizing row by row
```python

def tokenize_function(text_data):
    return tokenizer(text_data["text"], return_tensors="pt", padding=True, truncation=True, max_length=64)

# Tokenize in batches
tokenized_in_batches = train_data.map(tokenize_function, batched=True)

# Tokenize row by row
tokenized_by_row = train_data.map(tokenize_function, batched=False)
```
- Subword tokenization
    - Common in modern tokenizer
    - words split into meaningful sub-parts

## Fine-tuning through training
### Training Arguments
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./finetuned",
    evaluation_strategy="epoch", #epoch, steps, none
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01, # helps avoid overfitting
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_data,
    eval_dataset=tokenized_test_data,
    tokenizer=tokenizer
)

trainer.train()

# Using the fine-tuned model
new_data = ['new data here']

new_input = tokenizer(new_data, return_tensors="pt", padding=True, truncation=True, max_length=64)

with torch.no_grad():
    outputs = model(**new_input)

predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

label_map = {0: "NEGATIVE", 1: "POSITIVE"}
for i, predicted_label in enumerate(predicted_labels):
    sentiment = label_map[predicted_label]
    print(f"\nInput Text {i+1}: {new_data[i]}")
    print(f"Predicted Label: {sentiment}")

# Saving models and tokenizers
model.save_pretrained("my_finetuned_files")
tokenizer.save_pretrained("my_finetuned_files")

# Loading a saved model
model = AutoModelForSequenceClassification.from_pretrained("my_finetuned_files")
tokenizer = AutoTokenizer.from_pretrained("my_finetuned_files")
```

## Fine-tuning approaches
- Full fine-tuning
    - the entire model weights are updated
    - computationally expensive
- partial fine-tuning
    - some layers are fixed
    - only task-specific layers are updated

- Transfer learning
    - a pre-trained model is adapted to a different but related task
    - leverages knowledge from one domain to a related one
- N-shot learning
    - zero-shot learning: no examples
    - one-shot learning: one example
    - few-shot learning: several example
```python
# One-shot learninng
from transformers import pipeline

generator = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

input_text = """ input text here """

result = generator(input_text, max_length=100)
print(result[0]["label"])

```
# Evaluating LLM performance
- Text Classification
    - Accuracy
    - F1
- Text generation
    - Perplexity
    - BLEU
- Summarization
    - ROUGE score
    - BLEU score
- Translation
    - BLEU score
    - METEOR
- Question-answering
    - Exact Match (EM) F1 score: extractive QA
    - BLEU/ROUGE: generative QA
## The evaluate library
```python
import evaluate
accuracy = evaluate.load("accuracy")
accuracy.description
accuracy.features

f1 = evaluate.load("f1")
f1.features

pearson_corr = evaluate.load("pearsonr")
pearson_corr.features
```
### Classification metrics
```python
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

from transformers import pipeline

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

predictions = classifier(evaluation_text)

predicted_labels = [1 if pred["label"] == "POSITIVE" else 0 for pred in predictions]

real_labels = [0,1,0,1,1]
predicted_labels = [0,0,0,1,1]

print(accuracy.compute(references=real_labels, predictions=predicted_labels))
print(precision.compute(references=real_labels, predictions=predicted_labels) )
print(recall.compute(references=real_labels, predictions=predicted_labels) )
print(f1.compute(references=real_labels, predictions=predicted_labels) )
```
### Evaluating our fine-tuned model
```python
new_data = ["This is movie was disappointing!",
"This is the best movie ever!"]

new_input = tokenizer(new_data
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=64)

with torch.no_grad():
    outputs = model( ** new_input)

predicted = torch.argmax(outputs.logits,dim=1).tolist()

real = [0,1]
print(accuracy.compute(references=real,
predictions=predicted))
print(precision.compute(references=real,
predictions=predicted))
print(recall.compute(references=real,
predictions=predicted))
print(f1.compute(references=real,
predictions=predicted))
```

## Metrics for language tasks: perplexity and BLEU
### Perplexity
    - a model's ability to predict the next word accurately and confidently
    - lower perplexity = higher confidence
```python
input_text = ""
generated_text = ""

# Encode
input_text_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_text_ids, max_length=20)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)


perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(predictions=generated_text, model_id="gpt2")

print(results)
print(results["mean_perplexity"])
```

### BLEU
    - measures translation quality againts human references
    - predictions: LLMs outputs
    - references: human references
    - 0-1 score: closer to 1 = higher similarity
```python
bleu = evaluate.load("bleu")

input_text = ""
references = [["text1", "text2"]]
generated_text = ""

results = bleu.compute(predictions=[generated_text], references=references)
```
```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

# Translate the first input sentence then calucate the BLEU metric for translation quality
translated_output = translator(input_sentence_1)

translated_sentence = translated_output[0]['translation_text']

print("Translated:", translated_sentence)

results = bleu.compute(predictions=[translated_sentence], references=reference_1)
print(results)
```
```python
# Translate the input sentences, extract the translated text, and compute BLEU score
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

translated_outputs = translator(input_sentences_2)

predictions = [translated_output['translation_text'] for translated_output in translated_outputs]
print(predictions)

results = bleu.compute(predictions=predictions, references=references_2)
print(results)
```

## Metrics for language tasks: ROUGE, METEOR, EM
### ROGUE
- similarity between generated a summary and reference summaries
    - looks at n-grams and overlapping
    - ROUGE scores:
        - rouge1: unigram overlap
        - rouge2: bigram overlap
        - rougeL: long overlapping subsequences

### METEOR
- more linguistic feature like word variations, similar meanings, and word order
```python
results_bleu = bleu.compute (predictions=pred, references=ref)
results_meteor = meteor.compute(predictions=pred, references=ref)
print("Bleu: ", results_bleu['bleu' ])
print("Meteor: ", results_meteor['meteor'])
```

### Exact Match(EM)
- 1 if an LLM's output exactly matches its reference answer
- normally used in conjunction with F1 score
```python
from evaluate import load
em_metric = load("exact_match")

exact_match = evaluate.load("exact_match")
predictions = ["The cat sat on the mat.",
"Theaters are great.",
"Like comparing oranges and apples."]
references = ["The cat sat on the mat?",
"Theaters are great.",
"Like comparing apples and oranges."]

results = exact_match.compute(
references=references, predictions=predictions)
print(results)
```
## Safeguarding LLMs
### LLM challenges
- Multi-language support: language diversity, resource availability, adaptability
- Open vs closed LLMs dilemma: collaboration vs responsible use
- Model scalability: representation capabilities, computational demand, training requirements
- Biases: biased training data, unfair language understanding and generation

### Truthfulness and hallucinations
- Hallucinations: generated text contains false or nonsensical information as if it were accurate
### Metrics for analyzing LLM bias: toxicity
```python
toxicity_metric = load("toxicity")
texts_1 = ["Everyone in the team adores him", "He is a true genius, pure talent"]
texts_2 = ["Nobody in the team likes him", "He is a useless 'good-for-nothing' "]
toxicity_results_1 = toxicity_metric.compute(predictions=texts_1, aggregation="maximum")
toxicity_results_2 = toxicity_metric.compute(predictions=texts_2, aggregation="maximum")
print("Toxicity Sentences 1:", toxicity_results_1)
print("Toxicity Sentences 2:", toxicity_results_2)
```
```python
# Calculate the individual toxicities
toxicity_1 = toxicity_metric.compute(predictions=user_1)
toxicity_2 = toxicity_metric.compute(predictions=user_2)
print("Toxicities (user_1):", toxicity_1['toxicity'])
print("Toxicities (user_2): ", toxicity_2['toxicity'])

# Calculate the maximum toxicities
toxicity_1_max = toxicity_metric.compute(predictions=user_1, aggregation="maximum")
toxicity_2_max = toxicity_metric.compute(predictions=user_2, aggregation="maximum")
print("Maximum toxicity (user_1):", toxicity_1_max['max_toxicity'])
print("Maximum toxicity (user_2): ", toxicity_2_max['max_toxicity'])

# Calculate the toxicity ratios
toxicity_1_ratio = toxicity_metric.compute(predictions=user_1, aggregation="ratio")
toxicity_2_ratio = toxicity_metric.compute(predictions=user_2, aggregation="ratio")
print("Toxicity ratio (user_1):", toxicity_1_ratio['toxicity_ratio'])
print("Toxicity ratio (user_2): ", toxicity_2_ratio['toxicity_ratio'])
```

### Metrics for analyzing LLM bias: regard
- Regard: language polarity and biased perception towards certain demographics
```python
regard = load("regard")

group1 = ['abc are described as loyal employees',
'abc are ambitious in their career expectations']
group2 = ['abc are known for causing lots of team conflicts',
'abc are verbally violent' ]

polarity_results_1 = regard.compute(data=group1)
polarity_results_2 = regard.compute(data=group2)

for result in polarity_results_1['regard']:
    print(result)
```

```python
# Load the regard and regard-comparison metrics
regard = evaluate.load("regard")
regard_comp = evaluate.load("regard", "compare")

# Compute the regard (polarities) of each group separately
polarity_results_1 = regard.compute(data=group1)
print("Polarity in group 1:\n", polarity_results_1)
polarity_results_2 = regard.compute(data=group2)
print("Polarity in group 2:\n", polarity_results_2)

# Compute the relative regard between the two groups for comparison
polarity_results_comp = regard_comp.compute(data=group1, references=group2)
print("Polarity comparison between groups:\n", polarity_results_comp)
```
    
