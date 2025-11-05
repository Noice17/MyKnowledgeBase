# Understanding LLMs and Llama
## What is Llama
- A model that can do the ff locally:
    - Summarization
    - Data analysis
    - Coding assistant
### Why run Llama 3 locally?
- Cost efficiency
- Privacy and safety
- customization
- available locally when python is installed
- `pip install llama-cpp-python`
```python
from llama_cpp import Llama
llm = Llama(model_path = "path/to/model.gguf")
output = llm("Question here?")

output["choices"][0]["text"]
```
##  Tuning Llama 3 parameters
- Llama 3 decoding parameters
    - `Temperature`: controls randomness
        - 0-1
        - Low temperature (0): more predictable
        - High temperature (1): more creative response
    - `Top-K`: limits token selection to the most probable choices
        - Low k: more predictable
        - High k: more diverse response
    - `Top-P`: adjusts token selection based on cumulative probability
        - High top-p (close to 1): more varied responses
        - Low top-p (close to 0): less variation
    - `Max tokens`: limits response length
```python
llm - Llama()

output_concise = llm(
    "Describe an electric car.",
    temperature = 0.2,
    top_k = 1,
    top_p = 0.4,
    max_tokens = 20
)

output_creative = llm(
    "Describe an electric car.",
    temperature = 0.8,
    top_k = 10,
    top_p = 0.9,
    max_tokens = 100
)
```
## Assigning chat roles
- System role: sets the personality and style
- User role: represents the person asking the question

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# System role
system_message = "You are a business consultant who gives data-driven answers."

# User role
user_message = "What are the key factors in a successful marketing strategy?"

message_list = [
    {"role":"system", "content": system_message},
    {"role":"user", "content": user_message}
]

#Generating the response
response = llm.create_chat_completion(messages= message_list)
print(response)

# Assistant role
response["choices"][0]

result['choices'][0]['message']['content']
```
- 

# Using Llama Locally
## Guiding unstructure responses
- How to refine Llama's responses
    - Refine Prompts
    - Zero-shot/Few-shot prompting
        - Single instruction
        - Add labels
    - Use Stop Words
- Components of effective prompting
    - Precision
    - Avoid ambiguity
    - Use of keywords
    - Examples
    - Action words

## Generating Structured Output
- JSON responses with chat completion
```python 
response_format = {"type":"json_object"}


output = llm.create_chat_completion(
    messages = message_list
    response_format = "json_object:"
)
```

```python
output = llm.create_chat_completion(
    messages=messages,
    response_format={
        "type": "json_object",
        "schema": {
            "type": "object",
            # Set the properties of the JSON fields and their data types
            "properties": {"Question": {"type": "string"}, "Answer": {"type": "string"}}
        }
    }
)

print(output['choices'][0]['message']['content'])
```

## Building conversations
- Maintaining context
    - user inquiry
    - ai response
    - user follow-up
    - ai memory use
    - ai reponse
- Conversation class
    - can store a history of prior messages
```python
class Conversation:
    def __init__(self, llm: Llama, system_prompt='', history=[]):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = [{"role":"system", "content": self.system_prompt}] + history
    
    def create_completion(self, user_prompt=''):
        self.history.append({"role":"user", "content": user_prompt}) #Append input
        output = self.llm_create_chat_completion(messages=self.history)
        conversation_result = output['choices'][0]['message']
        self.history.append(conversation_result)
        return conversation_result['content']

# Running a multi-turn conversation
conversation = Conversation(llm, system_prompt="You are a virtual travel assistant helping with planning trips.")

response1 = conversation.create_completion("What are some destinations in France for a short weekend break?")

print(f"Response 1: {response1}")

response2 = conversation.create_completion("How about Spain?")

print(f"Response 2: {response2}")
```
