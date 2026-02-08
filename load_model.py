import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def load_model(model_path="models/climate_advisor_lora"):
    """Load LoRA model for agricultural advice"""
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # Load LoRA adapter
    if os.path.exists(model_path):
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print(f"Model path {model_path} not found, using base model")
        model = base_model
    
    model.eval()
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_tokens=500):
    """Generate response using loaded model"""
    formatted_prompt = f"<|system|>\nYou are an expert agricultural advisor.\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    device = next(model.parameters()).device
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    response = generate_response("What crops should I grow in Delhi?", model, tokenizer)
    print(response)