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

Sources: BSEE.pdf, BSCE.pdf, BSECE.pdf


Context:
[Source 1: BSCE.pdf]
The Eulogio 'Amang' Rodriguez Institute of Science and Technology (EARIST), under the College of Engineering, prescribes the following curriculum for the first two years of the Bachelor of Science in Civil Engineering (BSCE) program, effective June 2018.

**First Year - First Semester (Total: 29 Units)**
Students must complete the following foundational courses: 'CEENG111' (Civil Engineering Orientation, 2 units); 'CENMAT01' (Differential Calculus, 3 units); 'CENMATH 111' (Mathematics for Engineers, 3 units); 'GEARTAPP' (Art Appreciation, 3 units); 'GEKOMFIL' (Kontekstwalisadong Komunikasyon sa Filipino, 3 units); 'GEMATHMW' (Mathematics in the Modern World, 3 units); 'GEPEMOVE' (Movement Enhancement, 2 units); 'GESCIETS' (Science, Technology and Society, 3 units); 'NATSCI SP_102' (Chemistry for Engineers, 4 units); and 'NSTPROG1' (National Service Training Program 1, 3 units). No prerequisites are listed for this introductory semester.

**First Year - Second Semester (Total: 27 Units)**
The curriculum continues with: 'COM_SP_212' (Computer Fundamentals and Programming, 2 units); 'DRW 101A' (Engineering Drawing and Plans, 1 unit); 'GEPANIPI' (Panitikan sa Pilipinas / Philippine Literature, 3 units); 'GEPEFITE' (Fitness Exercises, 2 units); 'GEPURPCO' (Purposive Communication, 3 units); 'GEREADPH' (Readings in Philippine History, 3 units); 'GEUNDETS' (Understanding the Self, 3 units); 'MATH_104' (Calculus II / Integral Calculus, 3 units); 'NATSCI103A' (Physics for Engineers - Calculus Based, 4 units); and 'NSTPROG2' (National Service Training Program 2, 3 units).

**Second Year - First Semester (Total: 25 Units)**
Students advance to more technical subjects including: 'CAD 101' (Computer Aided Drafting, 1 unit); 'CVL 311A' (Fundamentals of Surveying, 4 units); 'DIFFEQUA' (Differential Equations, 3 units); 'EECO_311' (Engineering Economy, 3 units); 'GEADCPBR' (Communicative Proficiency in Business Correspondence and Research Writing, 3 units); 'GECONTWO' (The Contemporary World, 3 units); 'GELIFEWR' (The Life and Works of Rizal, 3 units); 'GEPEHEF1' (Physical Activity Towards Health and Fitness 1, 2 units); and 'STRIGBOD' (Statics of Rigid Bodies, 3 units). Notably, 'STRIGBOD' requires the completion of Calculus II and Physics for Engineers.

**Second Year - Second Semester (Total: 26 Units)**
The year concludes with: 'CENMECDB' (Mechanics of Deformable Bodies, 5 units); 'CVENG221' (Construction Materials and Testing, 3 units); 'CVENG222' (Engineering Data Analysis, 3 units); 'CVENG223' (Engineering Surveys, 3 units); 'CVENG224' (Geology for Civil Engineering, 2 units); 'GEADDPDS' (Practical Data Science, 3 units); 'GEETHICS' (Ethics, 3 units); 'GEPEHEF2' (Physical Activity Towards Health and Fitness II, 2 units); and 'MCH 322' (Dynamics of Rigid Bodies, 2 units).

[Source 2: BSEE.pdf]
The Eulogio 'Amang' Rodriguez Institute of Science and Technology (EARIST), under the College of Engineering, prescribes the following curriculum for the first two years of the Bachelor of Science in Electrical Engineering (BSEE) program. This curriculum is effective as of June 2018.

**First Year - First Semester (Total: 25 Units)**
The introductory semester consists of the following courses: 'CENCAD 111' (Computer Aided Drafting, 1 unit); 'CENMAT01' (Differential Calculus, 3 units); 'GEARTAPP' (Art Appreciation, 3 units); 'GEKOMFIL' (Kontekstwalisadong Komunikasyon sa Filipino, 3 units); 'GEMATHMW' (Mathematics in the Modern World, 3 units); 'GEPEMOVE' (Movement Enhancement, 2 units); 'GESCIETS' (Science, Technology and Society, 3 units); 'NATSCI_SP_102' (Chemistry for Engineers, 4 units); and 'NSTPROG1' (National Service Training Program 1, 3 units). No prerequisites are listed for this semester.

**First Year - Second Semester (Total: 27 Units)**
The curriculum continues with: 'GEELECIT' (Living in Information Technology Era, 3 units); 'GEPANIPI' (Panitikan sa Pilipinas, 3 units); 'GEPEFITE' (Fitness Exercises, 2 units); 'GEPURPCO' (Purposive Communication, 3 units); 'GEREADPH' (Readings in Philippine History, 3 units); 'GEUNDETS' (Understanding the Self, 3 units); 'MATH_104' (Calculus II / Integral Calculus, 3 units, Prereq: Differential Calculus); 'NATSCI103A' (Physics for Engineers - Calculus Based, 4 units, Prereq: Differential Calculus); and 'NSTPROG2' (National Service Training Program 2, 3 units).

**Second Year - First Semester (Total: 27 Units)**
Students advance to core engineering courses: 'CENPROG' (Computer Programming, 1 unit); 'DIFFEQUA' (Differential Equations, 3 units, Prereq: Calculus II); 'EEENG_211' (Electrical Circuits 1, 4 units, Prereq: Calculus II & Physics for Engineers); 'EEMCH 211' (Engineering Mechanics, 3 units, Prereq: Calculus II & Physics for Engineers); 'EEMEN 211' (Basic Thermodynamics, 2 units, Prereq: Physics for Engineers); 'GECONTWO' (The Contemporary World, 3 units); 'GEELECCP' (Communicative Proficiency in Business Correspondence and Research Writing, 3 units); 'GEETHICS' (Ethics, 3 units); 'GEPEHEF1' (Physical Activity Towards Health and Fitness 1, 2 units); and 'MM16' (Engineering Data Analysis, 3 units, Prereq: Calculus II).

**Second Year - Second Semester (Total: 22 Units)**
The second year concludes with: 'ECENG 221' (Electronics Circuits: Devices and Analysis, 4 units, Prereq: Electrical Circuits 1); 'EEENG 221' (Electrical Circuits 2, 4 units, Prereq: Electrical Circuits 1); 'EEENG 222' (Engineering Math for EE, 3 units, Prereq: Differential Equations); 'EEENG_223' (Electromagnetics, 2 units, Prereq: Physics for Engineers & Differential Equations); 'EEMCH 221' (Fundamentals of Deformable Bodies, 2 units, Prereq: Engineering Mechanics); 'EEMCH 222' (Fluid Mechanics, 2 units, Prereq: Physics 1); 'GEELECDS' (Practical Data Science, 3 units); and 'GEPEHEF2' (Physical Activity Towards Health and Fitness II, 2 units).

[Source 3: BSECE.pdf]
The Eulogio 'Amang' Rodriguez Institute of Science and Technology (EARIST), under the College of Engineering, prescribes the following curriculum for the first two years of the Bachelor of Science in Electronics and Communication Engineering (BSECE) program. This curriculum structure is effective as of June 2018. 

**First Year - First Semester (Total: 28 Units)**
The introductory semester consists of the following courses: 'CENCAD 111' (Computer Aided Drafting, 1 unit); 'CENMAT01' (Differential Calculus, 3 units); 'CENMATH 111' (Mathematics for Engineers, 3 units); 'GEARTAPP' (Art Appreciation, 3 units); 'GEKOMFIL' (Kontekstwalisadong Komunikasyon sa Filipino, 3 units); 'GEMATHMW' (Mathematics in the Modern World, 3 units); 'GEPEMOVE' (Movement Enhancement, 2 units); 'GESCIETS' (Science, Technology and Society, 3 units); 'NATSCI SP_102' (Chemistry for Engineers, 4 units); and 'NSTPROG1' (National Service Training Program 1, 3 units). No prerequisites are listed for this term.      

**First Year - Second Semester (Total: 29 Units)**
The curriculum continues with: 'CENENVIR' (Environmental Science and Engineering, 3 units); 'CENPHY2' (Physics 2, 4 units, Prereq: Chemistry); 'ECE 110L1' (ECE Workshop 1, 1 unit, Prereq: Math for Engineers); 'GEELECIT' (Living in Information Technology Era, 3 units); 'GEPANIPI' (Panitikan sa Pilipinas, 3 units); 'GEPEFITE' (Fitness Exercises, 2 units); 'GEREADPH' (Readings in Philippine History, 3 units); 'MATH 104' (Calculus II / Integral Calculus, 3 units, Prereq: Diff Calc & Math for Engineers); 'NATSCI103A' (Physics for Engineers - Calculus Based, 4 units); and 'NSTPROG2' (NSTP 2, 3 units).

**Second Year - First Semester (Total: 26 Units)**
Students advance to core engineering subjects: 'CENMSE' (Material Science and Engineering, 3 units); 'CENPROG' (Computer Programming, 1 unit); 'DIFFEQUA' (Differential Equations, 3 units, Prereq: Calculus II); 'ECENG211' (Electronics 1: Electronics Devices and Circuits, 4 units, Prereq: Physics for Engineers & ECE Workshop); 'EECO 311' (Engineering Economy, 3 units); 'EEENG 211' (Electrical Circuits 1, 4 units, Prereq: Physics 2); 'GEADCPBR' (Communicative Proficiency in Business Correspondence and Research Writing, 3 units); 'GEPEHEF1' (Physical Activity Towards Health and Fitness 1, 2 units); and 'MM16' (Engineering Data Analysis, 3 units).

**Second Year - Second Semester (Total: 28 Units)**
The second year concludes with: 'CVENG316' (Engineering Management, 2 units); 'ECENG221B' (Electronics 2: Electronic Circuit Analysis and Design, 4 units, Prereq: Electronics 1); 'ECENG222' (Communications 1: Principles of Communication System, 4 units, Coreq: Electronics 2); 'EEENG 221' (Electrical Circuits 2, 4 units, Prereq: Electrical Circuits 1); 'EEN 323' (Electromagnetics, 3 units, Prereq: Diff Eq); 'GEADDPDS' (Practical Data Science, 3 units); 'GEPEHEF2' (Physical Activity Towards Health and Fitness II, 2 units); 'GEUNDETS' (Understanding the Self, 3 units); and 'MATH319B' (Advance Engineering Mathematics for ECE, 3 units, Prereq: Diff Eq).

Question: What is the curriculum for first two years of the Bachelor of Science in Civil Engineering ? [/INST]"""

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