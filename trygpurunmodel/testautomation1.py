import pandas as pd
import json
import requests
import os
import time

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score  # Added for METEOR
    from rouge_score import rouge_scorer
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import numpy as np
    
    # Ensure all necessary NLTK resources are downloaded
    # METEOR specifically requires 'wordnet' and 'omw-1.4' for synonym/root matching
    resources = ['punkt', 'wordnet', 'omw-1.4', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)
        
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Metrics libraries missing ({e}). Install with: pip install nltk rouge-score transformers torch numpy")
    METRICS_AVAILABLE = False

class MetricsEvaluator:
    def __init__(self):
        self.available = METRICS_AVAILABLE
        if not self.available:
            return
            
        print("Initializing Metrics Evaluator (GPT-2 + METEOR + ROUGE)...")
        try:
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model.eval() 
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            self.smooth = SmoothingFunction().method1
        except Exception as e:
            print(f"Error loading metrics models: {e}")
            self.available = False

    def calculate(self, reference, hypothesis):
        # Default empty result
        res = {"BLEU": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0, "PPL": 0.0, "Burstiness": 0.0}
        
        if not self.available or not hypothesis.strip():
            return res

        try:
            # Tokenization
            ref_tokens = nltk.word_tokenize(reference) if reference else []
            hyp_tokens = nltk.word_tokenize(hypothesis)
            
            # 1. BLEU
            bleu = 0.0
            if ref_tokens and hyp_tokens:
                bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smooth)

            # 2. ROUGE-L
            rouge_l = 0.0
            if reference:
                scores = self.rouge_scorer.score(reference, hypothesis)
                rouge_l = scores['rougeL'].fmeasure

            # 3. METEOR (Uses wordnet for synonyms)
            meteor = 0.0
            if ref_tokens and hyp_tokens:
                meteor = meteor_score([ref_tokens], hyp_tokens)

            # 4. PPL & Burstiness
            ppl, burstiness = self._calculate_ppl_burstiness(hypothesis)
            
            # Final output dictionary
            return {
                "BLEU": round(bleu, 4),
                "ROUGE-L": round(rouge_l, 4),
                "METEOR": round(meteor, 4),
                "PPL": round(ppl, 2),
                "Burstiness": round(burstiness, 2)
            }
        except Exception as e:
            print(f"  Metrics calc error: {e}")
            return res

    def _calculate_ppl_burstiness(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids[:, :1024] if encodings.input_ids.size(1) > 1024 else encodings.input_ids

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            ppl = torch.exp(outputs.loss).item()
            
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return ppl, 0.0
            
        sent_ppls = []
        for sent in sentences:
            if len(sent.split()) < 4: continue
            s_ids = self.tokenizer.encode(sent, return_tensors="pt")
            if s_ids.size(1) > 1024: s_ids = s_ids[:, :1024]
            if s_ids.size(1) == 0: continue

            with torch.no_grad():
                loss = self.model(s_ids, labels=s_ids).loss
                sent_ppls.append(torch.exp(loss).item())
        
        burstiness = np.std(sent_ppls) if sent_ppls else 0.0
        return ppl, burstiness  


def run_test_automation(
    input_json_path,
    output_excel_path="test_results.xlsx",
    api_url="http://localhost:8000/chat/stream",
    limit=1
):
    print("Starting test automation with METEOR support...")
    evaluator = MetricsEvaluator()
    
    if not os.path.exists(input_json_path):
        print(f"Error: File not found at {input_json_path}")
        return

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    if limit:
        questions = questions[:limit]

    results = []
    print(f"Found {len(questions)} questions. Processing...")

    for i, item in enumerate(questions):
        question_id = item.get('id', i+1)
        question_text = item.get('question', '')
        ground_truth = item.get('ground_truth_answer', '')
        ground_truth_id = item.get('context_id', '')
        
        print(f"Processing Q{question_id}: {question_text[:50]}...")
        
        start_time = time.time()
        full_response = ""
        retrieved_chunks = []
        status = "Success"
        
        try:
            payload = {"message": question_text, "history": []}
            response = requests.post(api_url, json=payload, stream=True, timeout=120)
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            try:
                                data = json.loads(data_str)
                                if data.get("type") == "metadata":
                                    retrieved_chunks = data.get("chunks", [])
                                elif data.get("type") == "token":
                                    full_response += data.get("content", "")
                                elif data.get("type") == "done":
                                    break
                            except json.JSONDecodeError:
                                pass
            else:
                full_response = f"HTTP Error: {response.status_code}"
                status = "Error"
        except Exception as e:
            full_response = f"Error: {str(e)}"
            status = "Error"
            if "ConnectionError" in str(e): break

        duration = time.time() - start_time
        metrics = evaluator.calculate(ground_truth, full_response)
        
        # Retrieval tracking
        retrieved_ids = [str(chunk.get('id', '')) for chunk in retrieved_chunks]
        retrieval_success = str(ground_truth_id) in retrieved_ids if ground_truth_id else False

        results.append({
            "ID": question_id,
            "Question": question_text,
            "Ground Truth": ground_truth,
            "Model Answer": full_response,
            "Retrieval Success": 1 if retrieval_success else 0,
            "BLEU": metrics["BLEU"],
            "ROUGE-L": metrics["ROUGE-L"],
            "METEOR": metrics["METEOR"],  
            "Perplexity": metrics["PPL"],
            "Burstiness": metrics["Burstiness"],
            "Time (s)": round(duration, 2),
            "Status": status
        })

    if results:
        df = pd.DataFrame(results)
        df.to_excel(output_excel_path, index=False)
        print(f"Results saved to {output_excel_path}")

if __name__ == "__main__":
    # Update these paths as needed
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    dataset_path = os.path.join(project_root, "Dataset", "goldends", "100qds.json")
    output_path = os.path.join(base_dir, "automation_results.xlsx")
    
    # Set the number of questions to process (None for all)
    num_questions = 100
    run_test_automation(dataset_path, output_path, limit=num_questions)