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

Source: EARIST Curriculum 

Complete Bachelor of Fine Arts (Painting) Curriculum. This is the official Bachelor of Fine Arts (BFA) curriculum for the College of Architecture and Fine Arts at the Eulogio \"Amang\" Rodriguez Institute of Science and Technology (EARIST), effective June 2018.\n\n### Curriculum Overview\n\n| Year / Semester | Course Code | Course Description | Units (Lec/Lab) | Credit Units | Prerequisites | Corequisites |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n| **First Year - 1st Sem** | DRAWING1 | Drawing I | 1 / 2 | 3 | None | None |\n| | GEKOMFIL | Kontekstwalisadong Komunikasyon sa Filipino | 3 / 0 | 3 | None | None |\n| | GEPEMOVE | Movement Enhancement | 2 / 0 | 2 | None | None |\n| | GEPURCO | Purposive Communication | 3 / 0 | 3 | None | None |\n| | HISTWART | History of World Arts | 3 / 0 | 3 | None | None |\n| | MATRALS1 | Materials I (Modeling 1) | 1 / 2 | 3 | None | None |\n| | NSTPROG1 | National Service Training Program 1 | 3 / 0 | 3 | None | None |\n| | TECHNOS1 | Techniques I (Freehand Drawing 1) | 1 / 2 | 3 | None | None |\n| | VISPERC1 | Visual Perception 1 (Elements of Visual Arts) | 1 / 2 | 3 | None | None |\n| **First Year - 2nd Sem** | DRAFTPER | Drafting and Perspective | 1 / 2 | 3 | None | None |\n| | DRAWING2 | Drawing II | 1 / 2 | 3 | None | None |\n| | GEMATHMW | Mathematics in the Modern World | 3 / 0 | 3 | None | None |\n| | GEPANIPI | Panitikan sa Pilipinas / Philippine Literature | 3 / 0 | 3 | None | None |\n| | GEPEFTE | Fitness Exercises | 2 / 0 | 2 | None | None |\n| | MATRALS2 | Materials II (Modeling 2) | 1 / 2 | 3 | None | None |\n| | NSTPROG2 | National Service Training Program 2 | 3 / 0 | 3 | None | None |\n| | TECHNOS2 | Techniques II (Freehand Drawing 2) | 1 / 2 | 3 | None | None |\n| | VISPERC2 | Visual Perception 2 (Concept of Visual Org.) | 1 / 2 | 3 | None | None |\n| **Second Year - 1st Sem** | GEELECGS | Gender and Society | 3 / 0 | 3 | None | None |\n| | GEELECIT | Living in Information Technology Era | 2 / 1 | 3 | None | None |\n| | GEPEHEF1 | Physical Activity Towards Health and Fitness 1 | 2 / 0 | 2 | None | None |\n| | GESINEPP | Sinesosyedad/Pelikulang Panlipunan | 3 / 0 | 3 | None | None |\n| | GEUNDETS | Understanding the Self | 3 / 0 | 3 | None | None |\n| | MATRALS3 | Materials III (Concept, Techniques, Process) | 2 / 1 | 3 | None | None |\n| | TECHNQS3 | Techniques III (Color and Form 1) | 1 / 2 | 3 | None | None |\n| | VISSTUD1 | Visual Studies 1 (Composition 1) | 1 / 2 | 3 | None | None |\n| **Second Year - 2nd Sem** | GEARTAPP | Art Appreciation | 3 / 0 | 3 | None | None |\n| | GEELECCP | Communicative Proficiency in Business Correspondence and Research Writing | 3 / 0 | 3 | None | None |\n| | GEETHICS | Ethics | 3 / 0 | 3 | None | None |\n| | GEMATHMW | Mathematics in the Modern World | 3 / 0 | 3 | None | None |\n| | GEPEHEF2 | Physical Activity Towards Health and Fitness II | 2 / 0 | 2 | None | None |\n| | MATRALS4 | Materials IV (Continuation of Materials III) | 1 / 2 | 3 | None | None |\n| | ORIEARTS | Oriental Arts | 3 / 0 | 3 | None | None |\n| | TECHNQS4 | Techniques IV (Color and Form 2) | 1 / 2 | 3 | None | None |\n| | VISSTUD2 | Visual Studies 2 (Composition 2) | 1 / 2 | 3 | None | None |\n| **Third Year - 1st Sem** | ADVISTU1 | Advanced Visual Studies 1 (Advanced Composition 1) | 1 / 2 | 3 | None | None |\n| | ARTTHEO1 | Art Theory 1 | 1 / 2 | 3 | None | None |\n| | ARTWORK1 | Art Workshop 1 (Adobe Photoshop) | 1 / 2 | 3 | None | None |\n| | BTAPHOTO | Basic to Advance Photography | 1 / 2 | 3 | None | None |\n| | GECONTWO | The Contemporary World | 3 / 0 | 3 | None | None |\n| | GEREADPH | Readings in Philippine History | 3 / 0 | 3 | None | None |\n| | PAINTING1 | Painting 1 (Exploration and Analysis of Concept in Painting) | 1 / 2 | 3 | None | None |\n| | PHILARTS | Philippine Art | 3 / 0 | 3 | None | None |\n| **Third Year - 2nd Sem** | ADVISTU2 | Advanced Visual Studies 2 (Advanced Composition 2) | 1 / 2 | 3 | None | None |\n| | ARTTHEO2 | Art Theory 2 | 1 / 2 | 3 | None | None |\n| | ARTWORK2 | Art Workshop 1 (Corel Draw) | 2 / 1 | 3 | None | None |\n| | FASHDES | Costume & Fashion Design | 1 / 2 | 3 | None | None |\n| | GACULHER | Gallery Visit and Cultural Heritage Tour | 3 / 0 | 3 | None | None |\n| | GEELECDS | Practical Data Science | 3 / 0 | 3 | None | None |\n| | GESCIENTS | Science and Technology | 3 / 0 | 3 | None | None |\n| | PAINTING2 | Painting 2 (Portfolio Presentation) | 1 / 2 | 3 | None | None |\n| **Third Year - Summer** | COOPEDUC | Cooperative Education (200 hours)/On-the-Job Training | 1 / 2 | 3 | None | None |\n| **Fourth Year - 1st Sem** | ARTSEMR1 | Art Seminar 1 (Art Issues 1) | 1 / 0 | 3 | None | None |\n| | ARTWORK3 | Art Work 3 (Print Art) | 2 / 0 | 3 | None | None |\n| | GEELECES | Environmental Science | 3 / 0 | 3 | None | None |\n| | RESMETAR | Research Methods in the Arts | 2 / 0 | 5 | None | None |\n| **Fourth Year - 2nd Sem** | ARTSEMR2 | Art Seminar 2 (Art Issues) | 1 / 2 | 3 | None | None |\n| | ARTWORK4 | Art Workshop 4 (Installation Art and Mural Painting) | 2 / 1 | 3 | None | None |\n| | GELIFEWR | The Life and Works of Rizal | 3 / 0 | 3 | None | None |\n| | THESISPT | Thesis | 2 / 3 | 5 | None | None |\n\n**Total Number of Units:** 180\n**Effectivity Date:** June 2018

Question: What is the complete curriculum of BS Fine Arts? [/INST]"""

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