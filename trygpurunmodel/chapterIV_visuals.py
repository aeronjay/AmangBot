import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import glob

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Dataset", "Default AMBOT Knowledge Base"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "chapter_iv_visuals")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data Collection
data = []
all_text = ""
chunk_counts = {}

print(f"Scanning directory: {DATASET_DIR}")

# Walk through the directory to find all .json files
json_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.json"), recursive=True)

print(f"Found {len(json_files)} JSON files.")

if not json_files:
    print("No JSON files found! Please check the directory path.")
    exit()

for file_path in json_files:
    try:
        # Determine category from folder structure
        rel_path = os.path.relpath(file_path, DATASET_DIR)
        path_parts = rel_path.split(os.sep)
        
        if len(path_parts) > 1:
            current_category = path_parts[0]
        else:
            # For files in root, use the filename without extension
            current_category = os.path.splitext(path_parts[0])[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = json.load(f)
            
            # Handle if the json content is a list of chunks
            if isinstance(file_content, list):
                for chunk in file_content:
                    # Extract relevant fields
                    category = current_category
                    content = chunk.get('content', '')
                    
                    # Store for distribution analysis
                    if category not in chunk_counts:
                        chunk_counts[category] = 0
                    chunk_counts[category] += 1
                    
                    # Store for Word Cloud
                    all_text += " " + content
                    
                    # Store for Token Length analysis (approximate)
                    # Using a simple whitespace split + 1.3 multiplier to approximate token count
                    # or just word count if preferred. 
                    # For a more accurate count, one would use the specific tokenizer.
                    approx_tokens = len(content.split()) * 1.3 
                    
                    data.append({
                        'category': category,
                        'token_count': int(approx_tokens)
                    })
            else:
                # Handle single object json if exists (though most seem to be lists)
                pass
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

df = pd.DataFrame(data)

# --- Visual 1: Knowledge Base Distribution (Pie Chart) ---
print("Generating Figure 4.1: Distribution of Institutional Knowledge Base by Topic...")

# Aggregate top N categories, group others as 'Others' if too many
category_counts = df['category'].value_counts()
top_n = 8
if len(category_counts) > top_n:
    main_cats = category_counts[:top_n]
    others = pd.Series([category_counts[top_n:].sum()], index=['Others'])
    plot_data = pd.concat([main_cats, others])
else:
    plot_data = category_counts

plt.figure(figsize=(10, 8))
# Using a colormap that looks professional
colors = plt.cm.Paired(range(len(plot_data)))
plt.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Figure 4.1: Distribution of Institutional Knowledge Base by Topic', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_4.1_Knowledge_Base_Distribution.png'))
plt.close()


# --- Visual 2: Corpus Analysis (Word Cloud) ---
print("Generating Figure 4.2: Word Cloud Visualization...")

# Removing common stopwords - adding custom ones if necessary
stopwords = set([
    "the", "and", "of", "to", "in", "a", "is", "for", "on", "with", "as", "by", "an", 
    "are", "this", "that", "it", "be", "or", "from", "at", "which", "will", "may",
    "can", "has", "have", "not", "student", "students", "earist" # 'earist' and 'student' might be too common
])

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, 
                      background_color='white', 
                      stopwords=stopwords,
                      max_words=200,
                      contour_width=3, 
                      contour_color='steelblue').generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Figure 4.2: High-Frequency Terms in the Knowledge Base', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_4.2_Word_Cloud.png'))
plt.close()


# --- Visual 3: Chunk Sizes Distribution (Histogram) ---
print("Generating Figure 4.3: Distribution of Text Chunk Lengths...")

plt.figure(figsize=(10, 6))
plt.hist(df['token_count'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Figure 4.3: Distribution of Text Chunk Lengths for RAG Indexing', fontsize=14)
plt.xlabel('Token Count (Approximate)', fontsize=12)
plt.ylabel('Number of Chunks', fontsize=12)
plt.grid(axis='y', alpha=0.5)

# Add a vertical line for the mean/median
mean_tokens = df['token_count'].mean()
plt.axvline(mean_tokens, color='red', linestyle='dashed', linewidth=1)
plt.text(mean_tokens*1.05, plt.ylim()[1]*0.9, 'Mean: {:.0f}'.format(mean_tokens), color='red')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_4.3_Chunk_Length_Distribution.png'))
plt.close()

print(f"All visuals generated in '{OUTPUT_DIR}' directory.")
