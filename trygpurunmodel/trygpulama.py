import time
from llama_cpp import Llama

# 1. Initialize Model
llm = Llama(
    model_path="./Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=-1,      # Offload all layers to GPU
    n_batch=1024,          # Processes prompt in bigger batches
    n_ctx=6144,           # Context window
    verbose=False         # Logs off
)

# 2. The Full Prompt (System + 4 Chunks + Question)
prompt = """[INST] You are "AmangBot", a friendly and knowledgeable AI student assistant for EARIST (Eulogio "Amang" Rodriguez Institute of Science and Technology).

YOUR PERSONALITY:
- Be warm, approachable, and conversational - like a helpful senior student or advisor
- Use natural language and be encouraging
- If this is a follow-up question, acknowledge the connection to the previous topic naturally

RESPONSE STRUCTURE:
1. FIRST, directly answer the student's question based on the provided context sources
2. THEN, provide additional related information that might be helpful to the student

RESPONSE RULES:
1. ALWAYS base your answer ONLY on the provided context sources - never make up information
2. Start your direct answer with "According to [Source Name], ..." citing the specific source
3. If multiple sources are relevant, cite each one: "According to [Source 1], ... Additionally, [Source 2] states that..."
4. After answering the main question, add a section like "You might also find this helpful:" or "Related information:" to share additional relevant details from the sources (such as deadlines, requirements, procedures, tips, or related topics)
5. Use bullet points or numbered lists for multiple items, steps, or requirements
6. If the information is NOT in the provided context, respond: "I'm sorry, I don't have specific information about that in my current sources. You may want to check with the EARIST registrar or relevant office for the most accurate details."
7. End with a helpful follow-up when appropriate, like "Is there anything else you'd like to know about this?" or "Would you like more details about any specific part?"

Sources Available: EARIST Citizen's Charter 2025 (1st Edition)


Context from EARIST Sources:
[Source 1: EARIST Citizen's Charter 2025 (1st Edition)]
Category: EARIST Cavite Campus - Library Services
Topic: Special Library Services (Overnight, Referrals, Equipment)
The Cavite Campus Library offers several special services. Overnight or Weekend Use allows students to borrow books for home use by presenting a COR and signing a Book Card (10 minutes). The Issuance of Referrals provides students with a letter to visit other libraries; this requires a COR and signing a Referral Form List (10 minutes). Users can access Computers/Equipment by presenting a COR and signing a Log Sheet (10 minutes). Reservations for materials can be made via a Reservation Form (10 minutes). Finally, Visiting Users from other institutions or alumni can access the library by presenting a valid ID and a referral letter from their school (10 minutes).

[Source 2: EARIST Citizen's Charter 2025 (1st Edition)]
Category: Library Services
Topic: Circulation of Books - Students
Under Library Services, the Circulation of Books and Other Library Materials for students allows for the borrowing, lending, and returning of materials depending on availability. This service is classified as a Simple Government to Citizen (G2C) transaction with a total processing time of 9 minutes. To avail of this service, students must provide a Borrower's Slip, a copy of the Book Card, and their Student ID or a copy of their Certificate of Registration (COR). The process begins with the student searching for the topic in the Online Public Access Catalog (OPAC) and filling out a borrower's slip (3 minutes). If the material is found, the student proceeds to the charging desk to fill out the book card and submit their ID or COR (3 minutes). For returns, the student hands the material to the staff at the charging desk to retrieve their ID or COR (3 minutes).

[Source 3: EARIST Citizen's Charter 2025 (1st Edition)]
Category: Student Affairs and Services
Topic: Scholarship Application Process
Under the Student Affairs and Services, the Scholarship application process is a Simple G2C transaction available to students. The checklist of requirements includes a Certificate of Recognition secured from the Registrar's Office and a Report of Grades (ROG) secured from the Dean's or Registrar's Office. To avail of the service, the student applicant must submit PDF copies of the Certificate of Registration (COR) and ROG to the SARRMS Office Staff. The service is free of charge, though a fee of P20.00 applies if the documents are lost. The total processing time for the request is 1 day.

[Source 4: EARIST Citizen's Charter 2025 (1st Edition)]
Category: Student Registration and Records Management Services
Topic: Cross-Enrollment Procedure
Currently enrolled students may cross-enroll in another school if a subject is not offered at EARIST. The maximum allowance is six (6) units. The student must present a Letter of Intent (indicating school, subject, and schedule), Certificate of Registration (COR), and School ID to the Registrar. 

After evaluation of grades, the student pays a 120.00 PHP fee at the Cashier. The official receipt is submitted to the Registrar, who issues a Permit to Cross-Enroll in triplicate. This permit must be signed by the College Dean and the Registrar. Two copies are released to the student. The total processing time is approximately 14 minutes.

Student Question: How to get COR? [/INST]"""

print("Generating Response...\n" + "="*50)


# 3. Start Timer
start_time = time.time()

# 4. Run Generation with STREAM=TRUE
stream = llm(
    prompt, 
    max_tokens=1660, 
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