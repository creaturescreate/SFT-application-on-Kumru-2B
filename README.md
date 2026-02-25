## Supervised Fine-Tuning on a Language Model

This is a really easy supervised fine-tuning example using a labeled dataset, shaping an under-trained base language model into a chatbot with a personality. 

# Overview
This project applies **LoRA fine-tuning** to the chosen model, using a custom created JSONL dataset. The goal was to observe the training process and the outcome, turning a base model into a chatbot with a personality.

- Original dataset is not included. A sample dataset is given for demonstration is included in 'data/sample_data.jsonl'

# Model

The language model i prefered is **Kumru-2B-Base** by **VNGRS AI**(https://huggingface.co/vngrs-ai/Kumru-2B-Base)*. The reason i chose this language model is because it was more undertrained, so i could observe the changes after training more clearly, also the trained model reflects the given dataset personality better.

# How to Use
1. Train the model: 'python train.py'
2. Interact: 'python main.py'





