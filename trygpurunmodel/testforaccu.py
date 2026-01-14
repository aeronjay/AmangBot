import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig 
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
# Note: Newer ragas versions might prefer ragas.metrics.collections or other paths.
# Based on user warnings, trying to keep it simple or use the suggested path. 
# User warning: "Please use 'ragas.metrics.collections' instead"
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
except ImportError:
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall, context_precision

from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import ast
import os

# 1. Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "formatted_dataset_for_eval.csv")
df = pd.read_csv(csv_path)
df = df.head(10)

def parse_context(ctx):
    try:
        if isinstance(ctx, list): return ctx
        parsed = ast.literal_eval(ctx)
        return parsed if isinstance(parsed, list) else [str(ctx)]
    except:
        return [str(ctx)] 

df["contexts"] = df["contexts"].apply(parse_context)
df["reference"] = df["ground_truth"].astype(str)
dataset = Dataset.from_pandas(df[["question", "answer", "contexts", "reference"]])

# 2. Setup Local LLM 
model_path = os.path.normpath(os.path.join(script_dir, "..", "Models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1, # Still using your RTX 5050
    n_ctx=4096,      
    temperature=0,   
    verbose=False
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# 3. CONFIGURE SERIAL EXECUTION
# max_workers=1 ensures LlamaCpp only handles one request at a time
# timeout needs to be high for local LLM execution, especially on GPU/CPU mix
run_config = RunConfig(max_workers=1, timeout=1200)

# 4. Run Evaluation
print("Starting evaluation (Serial Mode)...")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=run_config # Use only this for concurrency control
)

# 5. Save Results
results_df = results.to_pandas()
results_df.to_csv(os.path.join(script_dir, "ragas_evaluation_results.csv"), index=False)
print("Done!")