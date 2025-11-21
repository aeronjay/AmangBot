from llama_cpp import Llama

llm = Llama(
    model_path="./Models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=-1,      # Offload all layers to GPU
    n_batch=4096,         # <--- CRITICAL: Processes prompt in one go (Speed Boost)
    n_ctx=4096,           # Match your model's context window
    verbose=False         # Logs off for production speed
)

# 2. Run a generation
prompt = f"""
<s>[INST]
You are **Amang Bot (Ambot)**, a precise and helpful university advisor. 
You must answer the user's question **using only the information provided in the INFORMATION SECTIONS**. 

### Your Answer Must Follow These Rules:
1. **Start with a direct answer** to the user's question.
2. After the direct answer, **provide additional relevant details**, background, or context based strictly on the provided information.
3. If the information sections **do not include the answer**, explicitly say:
   "The provided information does not contain the answer to this question."
4. Do NOT invent or assume facts that are not in the information sections.
5. Maintain a clear, formal, and student-friendly tone.

---

### INFORMATION SECTIONS:
1.4 No student shall be enrolled in more than one college to earn more than one academic degree at the same time.\n\n1.5 No student will be allowed to attend classes in any subject if his/her name is not included in the Official Class List issued by the Office of the Registrar. Otherwise, the Certificate of Registration shall be considered prior to final Official Class List.\n\n1.6 Cross enrollment in another school is allowed only upon the favorable endorsement by the Dean/Campus Administrator who shall recommend to the Registrar the issuance of the corresponding permit. A student may be allowed to cross-enroll in another school under the following conditions:\n\n1.1.1 The course/s is not offered in the Institute.\n1.1.2 The maximum number of units allowed for cross-enroll is six (6) units.\n1.1.3 The Registrar's Office shall prepare the cross-enrollment form to be carried by the student to the chosen school.\n1.1.4 The Registrar's Office should be provided a copy of his approved request.\n\n1.7 No cross enrollee from another institution shall be admitted into the Institute without the written permission from the Registrar's Office of the originating school. The permit shall state the total number of units and course description of the subject for which the student is authorized to take in the Institute.\n\n1.8 SECOND BACCALAUREATE DEGREE \nThe Institute does not accept applications for a second baccalaureate degree.

Students who did not meet the score requirements of their first-choice program will be advised to apply for admission in his/her second choice of program.\n\n\n## ADMISSION OF TRANSFEREES\n\nStudents may be admitted as transferees depending on the availability of slots provided that they meet all the academic and admission requirements set forth by the Institute as discussed in Section A.\n\nThe courses taken from the previous school of the student may be credited upon submission of Official Transcript of Records, to be evaluated by the Department Chairperson and the Registrar's Office.\n\nTransferees may be advised to use the old or the new curriculum depending on the credited courses.\n\nIf the old curriculum will be followed by the student, he/she does not need to take the Bridging Program for Lifelong Learners.\n\nIf the new curriculum will be followed by the student, he/she shall be required to enroll and pass the additional general education subjects identified in the Bridging Program for lifelong Learners presented in Section C.

##### 4.1.4.1 From EARIST Cavite Campus to Manila vice versa—Students may be admitted as transferees upon the recommendation of the EARIST Cavite Campus (ECC) Administrator, depending on the availability of slots and have complied with all the requirements for admission.\n\n##### 4.1.4.2 From Another School – Students may be admitted as transferees depending on the availability of slots and have met all the academic and admission requirements set forth by the Institute. The subject(s) taken from the previous school may be credited upon submission of Official Transcript of Records, to be evaluated by the Department Chairperson(s) and of the Registrar's Office using Form No. SARRMS C-1.\n\n#### 4.2 According to Academic Load\n\n##### 4.2.1 Regular student – One who is registered for formal academic credits and who carries full load required in a given semester in his/her curriculum.\n\n##### 4.2.2 Irregular student – One who is registered for formal credit but who carries less than full load required in a given semester in his/her curriculum.\n\n##### 4.2.3 Special student – One who is not earning formal academic credits or who is enrolled in the Special Opportunity Program.\n\n#### 4.3 Classification of Students According to Year Level\n\n##### 4.3.1 Freshmen – One who is in the first year of his curriculum or who has not finished the prescribed subjects of the first year of his curriculum, or has finished 25 percent of the total number of units required in his entire program.\n##### 4.3.2 Sophomore – One who is in the second year of his curriculum, or who has satisfactorily completed the prescribed subjects of the first year of his curriculum, or has finished more than 25 percent but less than 50 percent of the total number of units required in his entire program.\n##### 4.3.3 Junior – One who is in the third year of his curriculum, or who has completed the prescribed subjects of the first and second years of his curriculum, or has finished more than 50 percent but less than 75 percent of the total number of units required in his entire program.\n##### 4.3.4 Senior – One who is in the fourth year of his curriculum, or who has completed the prescribed subjects of the first, second, and third years of his curriculum, or has finished 75 percent or more of the total number of units required in his entire program.\n##### 4.3.5 Fifth – In the College of Engineering, Architecture, and Fine Arts, a student who is in the fifth year of his curriculum is considered a senior if he has completed at least 80 percent of the curricular requirement.

1.7 No cross enrollee from another institution shall be admitted into the Institute without the written permission from the Registrar's Office of the originating school. The permit shall state the total number of units and course description of the subject for which the student is authorized to take in the Institute.\n\n1.8 SECOND BACCALAUREATE DEGREE \nThe Institute does not accept applications for a second baccalaureate degree.\n\nSection 2. ACADEMIC LOAD\n\n2.1 A student shall carry the regular load every semester as prescribed in the curricular program.\n\n2.2 In no instance shall a student be allowed to carry less than one half of the regular load unless he has incurred academic deficiencies or is graduating during the semester, or the required courses are not offered during that semester.\n\n2.3 For regular terms, the maximum academic units for undergraduate students is the number of units indicated on the program checklist. A student may be allowed to enroll more than the maximum allowable load during a regular term under the following conditions:\n\n2.3.1 The overload during regular semester does not exceed 6 units over and above the regular load, provided that the overload courses are:\n- Back courses in their curriculum checklist and;\n- Courses without pre-requisites; and/or\n- Recommended by the Dean/Campus Administrator and approved by the Vice President for Academic Affairs.\n\n2.3.2 Only the regular load specified in the curriculum, considering the total number of training hours required. In no instance, a student may be allowed to carry an additional unit/s provided it is a non-major subject/s, not in conflict with the training schedule and not more than the allowed number of overload units.\n\n2.4 For the intersession class, the regular academic load is 6 units. A student may be allowed to enroll additional three (3) units under the following conditions:\n- Back courses in their curriculum checklist; and\n- Courses without pre-requisites.

Students who still have additional academic subjects currently enrolled (i.e., major and minor courses) are not allowed to have their training outside of the school except in specialized programs, provided the schedule of the enrolled subjects is outside the training hours. In such a case, the student shall accomplish a SARRMS form. Students who still have additional academic courses currently enrolled are allowed for in-house training in accordance with College requirements

---

### USER QUESTION:
Can I enroll in two different colleges at EARIST at the same time to get two degrees?

Provide your answer now, following the rules above.
[/INST]
"""

# Increased max_tokens so it can actually finish the sentence
output = llm(prompt, max_tokens=128, echo=False) 

print(output['choices'][0]['text'])