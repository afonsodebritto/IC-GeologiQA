import numpy as np

# Loading the dataset to train the model

from datasets import load_dataset

data = load_dataset (
    "jamescalam/image-text-demo",
    split="train"
)

# Initializing CLIP

from transformers import CLIPProcessor, CLIPModel
import torch

# Identification of the model
model_id = "openai/clip-vit-base-patch32"

# Creating the processor and the model for training
"""
The model is CLIP itself, but text and image data cannot be fed directly to it
The texts have to preprocessed to create "tokens IDs"
The images must be rezied and normalized
We use a processor that performs both of this functions before we feed CLIP with the training data
"""

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# Using GPU if possible, else just uses the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Encoding the text (creating the "tokens IDs")
text = data['text']
tokens = processor (
    text=text,
    padding=True,
    images=None,
    return_tensors='pt'
).to(device)

# This returns a dictonary with input_ids and attention_mask
# print(tokens.keys())
"""
input_ids are token ID values where each ID is an integer value that maps to a specific word or sub-word
For example the phrase "multi-modality" may be split into tokens ["multi", "-","modal", "ity"], that are mapped to IDs[1021, 11, 2427, 425]
Then, a text transformer maps these token IDs to semantic vector embeddings that the model learns on the pretraining phase
"""
"""
attention_mask is a tensor of 1s and 0s used by the model to "pay attention" to real token IDs and ignore padding tokens
Padding tokens are a special type of token used by text transformers to create input sequences of a fixed length from sentences of varying length. 
They are appended to the end of shorter sentences, so “hello world” may become “hello world [PAD] [PAD] [PAD]”.
"""

# Using CLIP to encode alll of these text descriptions
text_emb = model.get_text_features (
    **tokens
)

# These embeddings are not normalized, if we to use a similarity metric (like the dor product) we have to normalize the embeddings
# Not normalized:
"""
print(text_emb.shape)
print(text_emb.min(), text_emb.max())
"""

# Normalizing (if using dot product similarity)

# Detach text emb from graph, move to CPU and convert no np array
text_emb = text_emb.detach().cpu().numpy()

# Calculate value to normalize each vector
norm_factor = np.linalg.norm(text_emb, axis=1)
norm_factor.shape

text_emb = text_emb.T / norm_factor
# Transpose back to (21, 512)
text_emb = text_emb.T
"""
print(text_emb.shape)
print(text_emb.min(), text_emb.max())
"""

"""
If we didn't want to normalize the vectors, we could use cossine similarity as our metric
since it uses only angular similarity and not the vectors' magnitudes (like dot product).
In this code, the dot product will be used, so we need the normalization
"""

# Enconding the images (resizing and normalizing)
# We will be encoding the images using ViT
# print(data['image'][0].size)

# Creating the image batch
image_batch = data['image']
images = processor (
    text=None,
    images=image_batch,
    return_tensors='pt'
)['pixel_values'].to(device)

#print(images.shape)
"""
Preprocessing images consists of resizing the image to a 244x244 array with three color channels
RGB(red, green, blue) and normalizing pixel values into a [0,1][0,1] range
"""

# Getting the image features and normalizing then
img_emb = model.get_image_features(images)
"""
print(img_emb.shape)
print(img_emb.min(), img_emb.max())
"""

# Normalizing it
# Detach text emb from graph, move to CPU, and convert to numpy array
img_emb = img_emb.detach().cpu().numpy()
img_emb = img_emb.T / np.linalg.norm(img_emb, axis=1)
# Transpose back to (21, 512)
img_emb = img_emb.T
"""
print(img_emb.shape)
print(img_emb.min(), img_emb.max())
"""

# At this point we have sucessfully created CLIP embeddings for both text and images
# We can now compare items across the two modalities

# Calculating similarity
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Using cossine similarity
"""
cos_sim = np.dot(text_emb, img_emb.T) / (
    norm(text_emb, axis=1) * norm(img_emb, axis=1)
)

print(cos_sim.shape)
plt.imshow(cos_sim)
plt.show()
"""

# Using dot product
dot_sim = np.dot(text_emb, img_emb.T)
plt.imshow(dot_sim)
plt.show()