from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from collections import deque
import yaml

with open("config.yaml", "r"= as f:
    config = yaml.safe_load(f)

model_name = config["model"]["name"]
adapter_model_path = config["adapter"]["adapter_path"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
#-----------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
#-----------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, 
    device_map="auto")

#-----------------------------------------------------------------------------
print(f"Loading fine-tuned adapter from: {adapter_model_path}...")
model = PeftModel.from_pretrained(model, adapter_model_path)
print("Adapter loaded successfully!")
#-----------------------------------------------------------------------------
def complete_text(starting_text, long_term_memory="", short_term_memory="\n"):
    full_prompt = long_term_memory + short_term_memory + starting_text

    tokenized_text = tokenizer.encode_plus(full_prompt, return_tensors = 'pt', return_token_type_ids=False,
                                           max_length = model.config.max_position_embeddings, truncation = True).to(model.device)
    
    input_ids_length = tokenized_text['input_ids'].shape[1]

    max_new = 256
    max_total_length = min(model.config.max_position_embeddings, input_ids_length + max_new)

    generated_tokens = model.generate(**tokenized_text, 
                                      max_length = max_total_length,
                                      repetition_penalty = 1.1,
                                      no_repeat_ngram_size = 5,
                                      do_sample=True,
                                      temperature=config["tokens"]["temperature"],                 #Determines the randomness of the answers. keep between 1-0 where 1 is the most random.
                                      top_p=["tokens"]["top_p"],                         #Determines the variance in vocabulary. keep between 1-0 where 1 is the most various.
                                      pad_token_id=tokenizer.eos_token_id,
                                      eos_token_id=tokenizer.eos_token_id
                                     )
    new_tokens = generated_tokens[0][input_ids_length:]

    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    generated_text = generated_text.split(tokenizer.eos_token)[0]                

    return generated_text.strip()
  
# ---This is for an external long term memory in text format if it already exists --------------------------------------------
#----If there already exists a txt file, type 'memory' when you run the code if you'd like to add more to the memory file-----

def load_user_memory(filepath=config["data"]["external_memory"]):
    """Reads a text file and formats its content into a single prompt string."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            facts = f.read().strip()
            
        if not facts:
            return ""
        memory_prompt = (
            f"{facts}\n"
        )
        return memory_prompt
        
    except FileNotFoundError:
        print(f"Error: Couldn't find '{filepath}' . The model will not use an external memory.")
        return ""
      
#---------------------------------------------------------------------------
def save_user_memory(new_fact, filepath=config["data"]["external_memory"]):
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(new_fact + "\n")

memory_context = load_user_memory()
chat_history = deque(maxlen=8)
print("-"*40, "\nHello! Let's Chat!\nWrite 'exit' to leave the program\nWrite 'memory' to store data\n", "-"*40)   #<--------important after you run the main code

while True:
    starting_text = input("user: ")
    if starting_text.lower() in ["exit", "quit"]:
        break
    elif starting_text.lower() in ["clear memory", "clear mem", "cls mem"]:
        with open(config["data"]["external_memory"], 'w', encoding='utf-8') as f:
            f.write("")
        memory_context = load_user_memory()
        print("Memory cleared.")
        continue
    elif starting_text.lower() in ["memory", "mem"]:
        print("enter a prompt to keep in memory>>>")
        new_fact = input()
        save_user_memory(new_fact) 
        memory_context = load_user_memory() 
        print("Memory updated.")
        continue # <-- Added the missing 'continue'
    
    new_prompt_part = f"user: {starting_text}\nEbstheChatbot:"
    short_term_memory = "".join(chat_history)

    generated_text = complete_text(new_prompt_part, memory_context, short_term_memory)
     

    generated_text = generated_text.replace("</s>", "").strip() 

    print("EbstheChatbot: ", generated_text)
        
    full_exchange = f"user: {starting_text}\nEbstheChatbot: {generated_text}</s>\n"
    chat_history.append(full_exchange)

