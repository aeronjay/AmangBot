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

Sources Available: EARIST Curriculum June 2018, Student Handbook 2021


Context from EARIST Sources:
[Source 1: EARIST Curriculum June 2018]
Category: Curriculum - College of Computing Studies
Topic: Bachelor of Science in Computer Science
Comprehensive curriculum for the Bachelor of Science in Computer Science (BSCS) under the College of Computing Studies at EARIST. 

**Program Details**
* **Effectivity:** June 2018
* **Total Units:** 168

**FIRST YEAR**

**First Semester (26 units)**
* FPROGLAB Computer Programming 1 (Fundamentals of Programming) Lab (1 unit)
* FPROGLEC Computer Programming 1 (Fundamentals of Programming) Lec (2 units)
* GEARTAPP Art Appreciation (3 units)
* GEKOMFIL Kontekstwalisadong Komunikasyon sa Filipino (3 units)
* GELIFEWR The Life and Works of Rizal (3 units)
* GEMATHMW Mathematics in the Modern World (3 units)
* GEPEMOVE Movement Enhancement (2 units)
* GESCIETS Science, Technology and Society (3 units)
* NSTPROG1 National Service Training Program 1 (3 units)
* INTROCOMP Introduction to Computing (3 units)

**Second Semester (32 units)**
* DBMGTLAB Database Management System (Laboratory) (1 unit)
* DBMGTLEC Database Management System (Lecture) (2 units)
* DISSTRU1 Discrete Structure 1 (3 units). Prerequisite: DISSTRU1 (as stated in source)
* GEFILDIS Filipino sa Iba't ibang Disiplina (3 units)
* GEPANIPI Panitikan sa Pilipinas / Philippine Literature (3 units)
* GEPEFITE Fitness Exercises (2 units)
* GEPURPCO Purposive Communication (3 units)
* IPROGLAB Computer Programming 2 (Intermediate Programming) Laboratory (1 unit)
* IPROGLEC Computer Programming 2 (Intermediate Programming) Lecture (2 units). Prerequisite: CPROGLEC (Logic Formulation). Corequisite: IPROGLAB
* NSTPROG2 National Service Training Program 2 (3 units) (or NSTP-CWTS 2/LTS 2/MTS 2)

**SECOND YEAR**

**First Semester (20 units)**
* DIGDELAB Digital Logic Design (Laboratory) (1 unit)
* DIGDELEC Digital Logic Design (Lecture) (2 units)
* DISSTRU2 Discrete Structure 2 (3 units)
* DSALGLAB Data Structures and Algorithm (Laboratory) (1 unit)
* DSALGLEC Data Structures and Algorithm (Lecture) (2 units)
* GEELECCP Communicative Proficiency in Business Correspondence and Research Writing (3 units)
* GEPEHEF1 Physical Activity Towards Health and Fitness 1 (2 units)
* OOPRGLAB Object-Oriented Programming (Laboratory) (1 unit)
* OOPRGLEC Object-Oriented Programming (Lecture) (2 units)
* PHYCSLAB Physics for Computer Scientists (Laboratory) (1 unit)
* PHYCSLEC Physics for Computer Scientists (Lecture) (2 units)

**Second Semester (18 units)**
* CALCULUS Differential and Integral Calculus (3 units)
* GEPEHEF2 Physical Activity Towards Health and Fitness II (2 units)
* GPXVCLAB Graphics and Visual Computing (Laboratory) (1 unit)
* GPXVCLEC Graphics and Visual Computing (Lecture) (2 units)
* INFOMGMT Information Management (3 units)
* MICROBOT Microprocessor & Introduction to Robotics (1 unit)
* NETCOMMS Networks and Communications (3 units)
* PROGLLAB Programming Language (Laboratory) (1 unit)
* PROGLLEC Programming Language (Lecture) (2 units)

**THIRD YEAR**

**First Semester (19 units)**
* ALGCMPLX Algorithm and Complexity (3 units)
* APDEVLAB Application Development and Emerging Technology (Laboratory) (1 unit)
* APDEVLEC Application Development and Emerging Technology (Lecture) (2 units)
* ARCHIORG Architecture and Organization (3 units)
* HUCOMINT Human Computer Interaction (1 unit)
* INTELSYS Intelligent Systems (3 units)
* OPERSYST Operating System (3 units)
* SOFTENG1 Software Engineering 1 (3 units)

**Second Semester (18 units)**
* CSTHESI1 Thesis 1 (3 units)
* GEELECDS Practical Data Science (3 units)
* MOBAPLAB Mobile Application Development (iOS & Android) Laboratory (1 unit)
* MOBAPLEC Mobile Application Development (iOS & Android) Lecture (2 units)
* MOSIMLAB Modeling and Simulation (Laboratory) (1 unit)
* MOSIMLEC Modeling and Simulation (Lecture) (2 units)
* PARACOMP Parallel and Distributed Computing (3 units)
* SOFENGLA Software Engineering 2 (Laboratory) (1 unit)
* SOFTENG2 Software Engineering 2 (Lecture) (2 units)

**FOURTH YEAR**

**First Semester (20 units)**
* AUTOMATA Automata Theory and Formal Language (3 units)
* CSTHESI2 Thesis 2 (Lecture) (2 units)
* CSTHESL2 Thesis 2 (Laboratory) (1 unit)
* GECONTWO The Contemporary World (3 units)
* GEELECES Environmental Science (3 units)
* GEUNDETS Understanding the Self (3 units)
* INFOASEC Information Assurance and Security (2 units)
* SSUPPRAC Social Issues and Professional Practice (3 units)

**Second Semester (15 units)**
* CSINTERN Practicum/Internship (6 units)
* GEETHICS Ethics (3 units)
* GEREADPH Readings in Philippine History (3 units)
* INDOBSFT Industry Observation & Field Trips (3 units)

[Source 2: EARIST Curriculum June 2018]
Category: Curriculum - College of Engineering
Topic: Bachelor of Science in Computer Engineering
Comprehensive curriculum for the Bachelor of Science in Computer Engineering (BSCpE) under the College of Engineering at EARIST. 

**Program Details**
* **Effectivity:** June 2018 
* **Total Units:** 183 

**FIRST YEAR**

**First Semester (26 units)**
* CENCAD 111 Computer Aided Drafting (1 unit) 
* CENMAT01 Differential Calculus (3 units) 
* CENMATH 111 Mathematics for Engineers (3 units) 
* CPENG111 Computer Engineering as a Discipline (1 unit) 
* GEKOMFIL Kontekstwalisadong Komunikasyon sa Filipino (3 units) 
* GEPEMOVE Movement Enhancement (2 units) 
* GEREADPH Readings in Philippine History (3 units) 
* GEUNDETS Understanding the Self (3 units) 
* NATSCI SP_102 Chemistry for Engineers (4 units) 
* NSTPROG1 National Service Training Program 1 (3 units) 

**Second Semester (27 units)**
* CPE 112 Object Oriented Programming (2 units). Prerequisite: CPENG111 
* CPE 113 Programming Logic and Design (2 units) 
* GEFILDIS Filipino sa Iba't ibang Disiplina (3 units). Prerequisite: GEKOMFIL 
* GEMATHMW Mathematics in the Modern World (3 units) 
* GEPEFITE Fitness Exercises (2 units). Prerequisite: GEPEMOVE 
* MATH_104_5C Integral Calculus (5 units). Prerequisite: CENMAT01 
* MATH 122 Engineering Data Analysis (3 units). Prerequisite: CENMAT01 
* NATSCI103A Physics for Engineers (Calculus Based) (4 units). Prerequisite: NATSCI_SP_102 
* NSTPROG2 National Service Training Program 2 (3 units). Prerequisite: NSTPROG1 

**SECOND YEAR**

**First Semester (26 units)**
* CPENG211 Data Structures and Algorithm Analysis (2 units). Prerequisite: CPE 112, CPE 113 
* DIFFEQUA Differential Equations (3 units). Prerequisite: MATH_104_5C 
* EEENG211 Fundamentals of Electrical Circuits (4 units) 
* GEARTAPP Art Appreciation (3 units) 
* GEELECDS Practical Data Science (3 units). Prerequisite: MATH 122 
* GEELECIT Living in Information Technology Era (3 units) 
* GEETHICS Ethics (3 units) 
* GEPEHEF1 Physical Activity Towards Health and Fitness 1 (2 units) 
* GEPURPCO Purposive Communication (3 units) 

**Second Semester (25 units)**
* CENENECO Engineering Economy (3 units) 
* CPENG221 Numerical Methods (3 units). Prerequisite: DIFFEQUA 
* CPENG222 Software Design (4 units). Prerequisite: CPENG211 
* ECENG221 Fundamentals of Electronic Circuits (4 units) 
* GECONTWO The Contemporary World (3 units) 
* GELIFEWR The Life and Works of Rizal (3 units) 
* GEPEHEF2 Physical Activity Towards Health and Fitness II (2 units) 
* GESCIETS Science, Technology and Society (3 units) 

**THIRD YEAR**

**First Semester (24 units)**
* CENENVIR Environmental Science and Engineering (3 units). Prerequisite: NATSCI_SP_102 
* CPEELEC1 Cognate/Elective Course 1 (3 units) 
* CPENG311 Logic Circuits and Design (4 units). Prerequisite: ECENG221 
* CPENG312 Data and Digital Communications (3 units). Prerequisite: ECENG221 
* CPENG313 CPE Drafting & Design (1 unit). Prerequisite: CENCAD 111 
* CPENG314 Operating Systems (4 units). Prerequisite: CPENG222, CPENG211 
* CPENG315 Discrete Mathematics (3 units). Prerequisite: CPENG221 
* MXSIGSEN Fundamentals of Mixed Signal & Sensors (3 units). Prerequisite: ECENG221 

**Second Semester (21 units)**
* CENTECNO Technopreneurship (3 units) 
* CPEELEC2 Cognate/Elective Course 2 (3 units). Prerequisite: CPEELEC1 
* CPENG321 Basic Occupational Health & Safety (3 units) 
* CPENG322 Computer Network and Security (4 units). Prerequisite: CPENG312, CPENG314 
* CPENG323 Microprocessors (4 units). Prerequisite: CPENG311 
* CPENG324 CpE Laws and Professional Practice (2 units) 
* CVENG316 Engineering Management (2 units). Prerequisite: CENENECO 

**Summer (3 units)**
* CENOTJOB On-the-Job Training (300 hours) (3 units) 

**FOURTH YEAR**

**First Semester (20 units)**
* CPENG411 Methods of Research (2 units). Prerequisite: MATH_122, GEELECDS, CPENG311 
* CPENG412 Computer Architecture and Organization (4 units). Prerequisite: CPENG323 
* CPENG413 Embedded Systems (4 units). Prerequisite: CPENG323 
* CPENG414 Emerging Technologies in CpE (3 units) 
* CPENG415 CpE Practice and Design 1 (1 unit). Prerequisite: CPENG323, CPENG411 
* CPENG416 Feedback and Control System (3 units). Prerequisite: CPENG221, EEENG 221 (Electrical Circuits 2) 
* GEELECCP Communicative Proficiency in Business Correspondence and Research Writing (3 units) 

**Second Semester (11 units)**
* CPEELEC3 Cognate/Elective Course 3 (3 units). Prerequisite: CPEELEC2 
* CPENG421 CPE Practice and Design 2 (2 units). Prerequisite: CPENG415 
* CPENG422 Digital Signal Processing (4 units). Prerequisite: CPENG416 
* CPENG423 Introduction to HDL (1 unit). Prerequisite: CPENG311 
* CPENG424 Seminars and Field Trips (1 unit) 

[Source 3: Student Handbook 2021]
Category: Academic Policies
Topic: Complete Curricular Offerings (Main and Cavite Campuses)
The EARIST Student Handbook 2021 details the complete curricular offerings for both the Main Campus and the Cavite Campus. 

**Main Campus Offerings by College:**

* **College of Architecture and Fine Arts (CAFA):** Bachelor of Science in Architecture (BS ARCHI), Bachelor of Science in Interior Design (BSID), and Bachelor of Fine Arts (BFA) with majors in Painting or Visual Communication.
* **College of Arts and Sciences (CAS):** Bachelor of Science in Applied Physics with Computer Science Emphasis (BSAP), Bachelor of Science in Psychology (BSPSYCH), and Bachelor of Science in Mathematics (BSMATH).
* **College of Computing Studies (CCS):** Bachelor of Science in Computer Science (BSCS), Bachelor of Science in Information Technology (BS INFO. TECH.)
* **College of Business and Public Administration (CBPA):** Bachelor of Science in Office Administration (BSOA), Bachelor of Science in Business Administration (Majors: Human Resource Management, Marketing Management), Bachelor of Science in Entrepreneurship (BSEntrep), and Bachelor in Public Administration (BPA).
* **College of Education (CED):** Bachelor in Secondary Education (BSE) (Majors: Filipino, Mathematics, Science), Bachelor of Technology and Livelihood Education (BTLEd) (Majors: Home Economics, Industrial Arts), Bachelor of Special Needs Education (BSNEd), and Professional Education Subjects (TCP - 18 units).
* **College of Engineering (CEN):** Bachelor of Science in Chemical Engineering (BSChE), Civil Engineering (BSCE), Electrical Engineering (BSEE), Electronics and Communication Engineering (BSECE), Mechanical Engineering (BSME), and Computer Engineering (BSCoE).
* **College of Hospitality and Tourism Management (CHTM):** Bachelor of Science in Tourism Management (BST) and Hospitality Management (BSHM).
* **College of Industrial Technology (CIT):** Bachelor of Science in Industrial Technology (BSIT) with majors in Automotive Technology, Electrical Technology, Electronics Technology, Food Technology, Fashion Apparel Technology, Industrial Chemistry, Drafting Technology, Machine Shop Technology, and Refrigeration and Air-conditioning.
* **College of Criminal Justice Education:** Bachelor of Science in Criminology (BSCrim).
* **Graduate School:** Master of Science in Mathematics, Master of Arts in Industrial Psychology, Master in Business Administration, Master in Public Administration, Master of Arts in Industrial Education (Major: Hotel Management), Master of Arts in Education (Majors: Administration and Supervision, Guidance & Counseling, Special Education), Master of Arts in Teaching (Majors: Electronics Technology, Mathematics, Science), Doctor of Philosophy in Industrial Psychology, Doctor of Education in Educational Management, Doctor in Business Administration, and Doctor in Public Administration.

**EARIST Cavite Campus Offerings:**

* **Graduate Program:** Doctor in Education (Major: Educational Management), Master of Arts in Education (Major: Administration and Supervision), and Master in Business Administration.
* **Post Baccalaureate:** Professional Education Subjects (18 units).
* **Undergraduate Program:** Bachelor of Science in Business Administration (Major: Marketing Management), Bachelor of Science in Computer Science, Bachelor of Science in Computer Technology, Bachelor of Science in Criminology, Bachelor of Science in Hospitality Management, Bachelor of Science in Office Administration, Bachelor of Science in Industrial Psychology, Bachelor of Technology and Livelihood Education, and Bachelor of Science in Industrial Technology (Majors: Automotive, Electrical, Electronics, Food Technology, Drafting Technology).

Student Question: What is the curriculum for BSCS? [/INST]"""

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