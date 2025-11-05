"""
Interactive Mistral-7B-Instruct for Student Queries
Optimized for RTX 5050 8GB VRAM using 4-bit quantization
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🎓 Student Query Assistant - Mistral 7B Interactive Mode")
print("=" * 70)

# Check GPU availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU Detected: {gpu_name}")
    print(f"💾 VRAM Available: {gpu_memory:.1f}GB")
else:
    print("⚠️  No GPU detected. Running on CPU (will be slower)")

# Configure 4-bit quantization for efficient memory usage
print("\n🔧 Configuring 4-bit quantization for optimal VRAM usage...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Mistral 7B model
print("\n🤖 Loading Mistral-7B-Instruct model...")
print("   (This may take a minute on first run...)")

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print(f"✅ Model loaded successfully: {model_name}")
    
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"📊 GPU Memory Used: {allocated_memory:.2f}GB / {gpu_memory:.1f}GB")
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("\n💡 Tip: Make sure you have:")
    print("   1. Hugging Face account and token")
    print("   2. Access to Mistral model (accept license on HuggingFace)")
    print("   3. Run: pip install transformers accelerate bitsandbytes")
    exit(1)

def generate_answer(question, max_new_tokens=350):
    """Generate answer for student query"""
    
    # Create optimized prompt for student queries
    prompt = f"""<s>[INST] You are a helpful university student advisor. Answer the student's question clearly and concisely.

STUDENT QUESTION: {question}

Provide a clear, helpful answer. Be specific and practical in your advice. [/INST]"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,  # Balanced creativity
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract generated answer
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = response.find("[/INST]") + len("[/INST]")
    answer = response[answer_start:].strip()

    return answer

def interactive_mode():
    """Interactive mode for asking student questions"""
    print("\n" + "=" * 70)
    print("🎓 Interactive Student Query Assistant")
    print("=" * 70)
    print("\n📝 Instructions:")
    print("   • Type your question and press Enter")
    print("   • Type 'quit', 'exit', or 'q' to exit")
    print("   • Type 'clear' to clear the screen")
    print("-" * 70)

    while True:
        print("\n" + "─" * 70)
        question = input("❓ Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the Student Query Assistant!")
            print("   Goodbye!")
            break

        if question.lower() == 'clear':
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🎓 Interactive Student Query Assistant")
            print("-" * 70)
            continue

        if not question:
            print("⚠️  Please enter a question!")
            continue

        try:
            print("\n🤔 Generating answer...\n")
            answer = generate_answer(question)
            
            print("💡 Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)
            
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(0) / 1e9
                print(f"📊 GPU Memory: {memory_used:.2f}GB")
            
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            print("   Please try rephrasing your question or restart the script.")

# Start interactive mode
if __name__ == "__main__":
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
