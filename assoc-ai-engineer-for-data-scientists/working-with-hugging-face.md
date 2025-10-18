# Getting Started with Hugging Face
## Running Hugging Face models
```python
from transformers import pipeline

gpt2_pipeline = pipeline(task = "text-generation", model = "openai-communitygpt2" )

print(gpt2_pipeline("What if AI", max_new_tokens = 10, num_return_sequences=2))

for result in results:
    print(result['generated_text'])
```
### Using inference providers
```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider = "together",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model = "deepseek-ai/DeepSeek-V3,
    messages = [
        {
            "role":"user",
            "content":"What is the capital of France?"
        }
    ]
)
print(completion.choices[0].message)
```
## Hugging Face Datasets
```python
from datasets import load_dataset

data = load_dataset("IVN-RIN/BioBERT_Italian")


data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")
```

### Apache Arrow dataset formats
```python
data = load_dataset("IVN-RIN/BioBERT_Italian", split="train")

filtered = data.filter(lambda row: " bella " in row["text"])
print(filtered)


# Select the first two rows

sliced = filtered.select(range(2))
print(sliced)

print(sliced[0]['text'])
```

# Building Pipelines with Hugging Face
## Text Classification
### Sentiment analysis
- labels text based on its emotional tone
```python
from transformers import pipeline

my_pipeline = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print(my_pipeline("Wi-Fi is slower than a snail today"))
```
### Grammatical correctness
- evaluates grammer for correctness

```python
from transformers import pipeline
grammar_checker = pipeline(
    task="text-classification",
    model="abdulmatinomotoso/English_Grammar_Checker"
)

print(grammar_checker("He eat pizza every day."))
```

### QNLI
- checks if a premise answers a question
```python
from transformers import pipeline
classifier = pipeline(
    task="text-classification",
    model="cross-encoder/qnli-electra-base"
)

classifier("Where is Seattle located?, Seattle is located in Washinton state")
```

### Dynamic category assignment
- dynamically assigns categories based on content
```python
from transformers import pipeline
classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli"
)

text  = "asdads"
categories = ["marketing","sales", "support"]

output = classifier(text, categories)
print(f"Top Label: {output['labels'][0]} with score: {output['scores'[0]]}")
```

## Text Summarization
- extractive
    - selects key sentences from the text
    - efficient, needs fewer resources
    - lacks flexibility; may be less cohesive

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="nyamuda/extractive-summarization")

text = "This is my really large text about Data Sciece..."
summary_test = summarizer(text)

print(summary_text[0]['summary_text'])
```
- abstractive
    - generates new, rephrased text
    - clearer and more readable
    - requires more resources and processing
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text = "This is my really large text about Data Sciece..."
summary_test = summarizer(text)

print(summary_text[0]['summary_text'])
```

### Parameters for summarization
- min_new_tokens & max_new_tokens: Controls summary length
```python
summarizer = pipeline(task="summarization", min_new_tokens=10, max_new_tokens=150)
```
## Auto Models and Tokenizers
- flexible access to models and tokenizers
- more control over model behavior and outputs
- perfect for advanced tasks
- pipelines = quick ; auto classes = flexible

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
```
### AutoTokenizers
- prepare text input data
- recommended to use the tokenizer paired with the model

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokens = tokenizer.tokenize("AI: Helping robots think and humans overthink :)")
print(tokens)
```
### Building a Pipeline with Auto Classes
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

my_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

my_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

my_pipeline = pipeline(
    task = "sentiment-analysis", 
    model = my_model, 
    tokenizer = my_tokenizer
)
```

```python
# Download the model and tokenizer
my_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
my_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Create the pipeline
my_pipeline = pipeline(task="sentiment-analysis", model=my_model, tokenizer=my_tokenizer)

# Predict the sentiment
output = my_pipeline("This course is pretty good, I guess.")
print(f"Sentiment using AutoClasses: {output[0]['label']}")
```
## Document Q&A
- answers question from document content
- requires a document and a question
- provides direct or paraphrased answers

```python
from pypdf import PdfReader

reader = PdfReader("US-Employee_Policy.pdf")

document_text = ""

for page in reader.pages:
    document_text += page.extract_text()


# Q&A pipeline

qa_pipeline = pipeline(
    task = "question-answering",
    model = "distilbert-base-cased-distilled-squad"
)

question = "How many volunteer days are offered annually?"

result = qa_pipeline(question = question, context = document_text)

print(f"Answer: {result['answer']}")
```