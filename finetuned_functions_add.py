# Add these functions after line 412 in app.py

def load_finetuned_model(model_path="models/climate_advisor_finetuned"):
    """Load fully finetuned model for state-wise crop recommendations"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import os
        
        if not os.path.exists(model_path):
            return None, None
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        model.eval()
        st.success("✅ Finetuned model loaded!")
        return model, tokenizer
        
    except ImportError:
        return None, None
    except Exception as e:
        st.warning(f"⚠️ Finetuned model failed: {str(e)}")
        return None, None

def generate_with_finetuned(prompt: str, model, tokenizer, max_tokens: int = 200) -> Optional[str]:
    """Generate response using fully finetuned model"""
    if model is None or tokenizer is None:
        return None
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Format prompt
        formatted_prompt = f"<|system|>\nYou are an agricultural expert.\n
