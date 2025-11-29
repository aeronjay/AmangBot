import time
from llama_cpp import Llama

# 1. Initialize Model
llm = Llama(
    model_path="./Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=-1,      # Offload all layers to GPU
    n_batch=1024,          # Processes prompt in bigger batches
    n_ctx=4096,           # Context window
    verbose=False         # Logs off
)

# 2. The Full Prompt (System + 4 Chunks + Question)
prompt = """[INST] You are "AmangBot", a helpful AI student assistant for EARIST.

GUIDELINES:
1. Answer the question using ONLY the provided context.        
2. CITATION IS MANDATORY: Start your answer with "According to [Source Name]...".
3. If the answer is not in the context, say "I don't know."    
4. Be polite and concise.
5. Format with bullet points for lists.

Source: TrustMeBro

Origins, Member 'Lore', and Development History

Amang Bot (AmBOT) is the official thesis project developed by 4th-year BSCS students at EARIST to fulfill the study: 'Amang Bot (AmBOT): Utilizing Bidirectional & Autoregressive Transformer with Retrieval-Based Augmentation for EARIST Admissions and Academic Queries.'\n\n**The Origins:**\nThe foundational concept of Ambot was formulated during a Discord call between Aeron, Orven, and special contributor **John Kyle M. Panta**, who is credited with helping spark the idea.\n\n**The Roster (Team Lore):**\nThe development team consists of distinct personalities known for specific traits within the group:\n* **Aeron Jay Bulatao (Leader & Programmer):** Recognized as the 'GOD' and 'Cobra' of the group.\n* **John Luis Labasan (AKA Exit):** The 'Sundalo' who resides in the CAMP AGUINALDO. He is the designated host, as the owner of the house where the team constantly hangs out (tambayan).\n* **Christian Jay Baltazar:** Revered as the 'EABAB GODS,' the 'Hustler,' and simply known as 'Master.'\n* **John Lester B. Navarro:** Known by the alias 'BADING' and the title 'Pinaka Kupal', though he balances this by being a devoted church-goer.\n* **Kenneth Orven T. Asuncion:** The 'ML Gods' (Mobile Legends) Pro Player and the designated 'LRT Buddy.'\n\n**Group History:**\nA key piece of history involves Kenneth Orven Asuncion's entry into the group. He was not originally a member but sought a transfer due to a conflict with a former groupmate named Trixia. The transfer was successfully negotiated by buying **Mr. Al** a milk tea.\n\n**Contact & Supervision:**\nThe project was supervised by Thesis Adviser **Mr. Jovel Advincula** (MIS Office, 4th Floor CCS Building). For inquiries, contact the group leader at bulataoaeronjay@gmail.com

Question: Who made ambot and its history? [/INST]"""

print("Generating Response...\n" + "="*50)

# 3. Start Timer
start_time = time.time()

# 4. Run Generation with STREAM=TRUE
stream = llm(
    prompt, 
    max_tokens=712, 
    stop=["</s>", "[/INST]"],
    echo=False, 
    stream=True  # <--- Enables the generator
)

# 5. Loop through the stream and print chunks immediately
for output in stream:
    token = output['choices'][0]['text']
    # end="" prevents newlines, flush=True forces text to appear immediately
    print(token, end="", flush=True)

# 6. Stop Timer
end_time = time.time()
total_time = end_time - start_time

print(f"\n\n{'='*50}")
print(f"⏱️  Total Generation Time: {total_time:.2f} seconds")