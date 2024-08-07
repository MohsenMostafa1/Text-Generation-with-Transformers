# Text-Generation-with-Transformers

<figure>
        <img src="https://miro.medium.com/v2/resize:fit:672/1*ET54ajLE5REySpe-5HDS8A.png" alt ="Audio Art" style='width:800px;height:500px;'>
        <figcaption>

#### Text generation is a pivotal aspect of AI, particularly within the field of natural language processing (NLP). It refers to the automated process of producing coherent and contextually relevant text that mimics human writing. This capability has evolved significantly, leveraging advancements in machine learning and deep learning technologies.  

#### Importing libraries and Loading the Pre-trained Model and Tokenizer.
Importing torch: PyTorch library for tensor operations and model handling, and LlamaForCausalLM: Class for the causal language model specific to LLaMA, and LlamaTokenizer: Tokenizer for encoding and decoding text. 
The model and tokenizer are loaded using the from_pretrained method:
model_name: Specifies the pre-trained LLaMA model to be used.
LlamaForCausalLM.from_pretrained: Loads the pre-trained model.
LlamaTokenizer.from_pretrained: Loads the corresponding tokenizer.

 ```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "openlm-research/open_llama_3b"  
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
```
#### Setting the Device and Defining the Prompt
The script checks for the availability of a GPU and sets the device accordingly annd prompt is defined as the starting point for text generation::
torch.cuda.is_available(): Checks if a GPU is available.
torch.device: Specifies the device (CPU or GPU).
model.to(device): Moves the model to the specified device.
prompt: A string containing the initial text.

 ```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
prompt = "Once upon a time in a faraway land, there was a small village where people lived in harmony with nature. One day,"
```

#### Encoding the Input Prompt and Generating Text
tokenizer.encode: Converts the prompt to token IDs, and return_tensors="pt": Returns the token IDs as PyTorch tensors, and to(device): Moves the tensors to the specified device.
max_length: Maximum length of the generated text, and model.generate: Generates text sequences, and num_return_sequences: Number of text sequences to generate.

 ```python
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
max_length = 200
outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
```

#### Decoding the Generated Text and Printing the Generated Text.
tokenizer.decode: Converts token IDs back to text and skip_special_tokens=True: Removes special tokens from the output.

 ```python
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
```

#### The script demonstrates how to use a pre-trained LLaMA model for text generation. It covers loading the model and tokenizer, preparing the input, generating text, and decoding the output. This process can be adapted to various text generation tasks by modifying the prompt and parameters.




