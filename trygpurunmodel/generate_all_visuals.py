import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Options for plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Dataset", "Default AMBOT Knowledge Base"))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "automation_results no 2.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "chapter_iv_visuals_complete")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_eda_visuals():
    print("--- Generating EDA Visuals (Section 4.1 & 4.2) ---")
    
    data = []
    all_text = ""
    chunk_counts = {}
    
    # 1. Load Data
    print(f"Scanning directory: {DATASET_DIR}")
    json_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.json"), recursive=True)
    
    if not json_files:
        print("No JSON files found!")
        return

    for file_path in json_files:
        try:
            # Determine category
            rel_path = os.path.relpath(file_path, DATASET_DIR)
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) > 1:
                category = path_parts[0]
            else:
                category = os.path.splitext(path_parts[0])[0]

            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
                if isinstance(content, list):
                    for chunk in content:
                        text = chunk.get('content', '')
                        if not text: continue
                        
                        all_text += " " + text
                        # Estimate tokens (approx 1.3 tokens per word is a common heuristic, or just word count)
                        word_count = len(text.split())
                        
                        data.append({
                            'category': category,
                            'word_count': word_count
                        })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    df = pd.DataFrame(data)
    
    # Visual 1: Knowledge Base Distribution (Pie Chart)
    print("Generating Figure 4.1: Knowledge Base Distribution...")
    plt.figure(figsize=(10, 8))
    category_counts = df['category'].value_counts()
    
    # Group small categories
    top_n = 8
    if len(category_counts) > top_n:
        main_cats = category_counts[:top_n]
        others = pd.Series([category_counts[top_n:].sum()], index=['Others'])
        plot_data = pd.concat([main_cats, others])
    else:
        plot_data = category_counts
        
    colors = sns.color_palette('pastel')[0:len(plot_data)]
    plt.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Distribution of Institutional Knowledge Base by Topic')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4.1_KB_Distribution.png'), dpi=300)
    plt.close()

    # Visual 2: Word Cloud
    print("Generating Figure 4.2: Word Cloud...")
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(all_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Terms in Knowledge Base')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4.2_WordCloud.png'), dpi=300)
    plt.close()

    # Visual 3: Chunk Length Distribution
    print("Generating Figure 4.3: Chunk Length Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Text Chunk Lengths')
    plt.xlabel('Word Count per Chunk')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4.3_ChunkLength_Distribution.png'), dpi=300)
    plt.close()
    
    print("EDA Visuals Complete.\n")

def generate_evaluation_visuals():
    print("--- Generating Evaluation Visuals (Section 4.3 & 4.4) ---")
    
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found at {RESULTS_FILE}")
        return

    try:
        df = pd.read_csv(RESULTS_FILE)
        # Ensure numeric columns
        df['Accuracy Score'] = pd.to_numeric(df['Accuracy Score'], errors='coerce')
        df['Response Latency (Seconds)'] = pd.to_numeric(df['Response Latency (Seconds)'], errors='coerce')
        
        # Clean Labels
        df['Complexity Level'] = df['Complexity Level'].str.capitalize()
        df['Query Type'] = df['Query Type'].str.capitalize()

        # Visual 4: Accuracy by Complexity
        print("Generating Figure 4.4: Accuracy by Complexity...")
        plt.figure(figsize=(8, 6))
        stats_acc = df.groupby('Complexity Level')['Accuracy Score'].mean().reset_index()
        sns.barplot(x='Complexity Level', y='Accuracy Score', data=stats_acc, palette='viridis')
        plt.ylim(0, 1.05)
        plt.title('Mean Accuracy by Question Complexity')
        plt.ylabel('Average Accuracy Score')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4.4_Accuracy_by_Complexity.png'), dpi=300)
        plt.close()

        # Visual 5: Latency by Complexity
        print("Generating Figure 4.5: Latency by Complexity...")
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Complexity Level', y='Response Latency (Seconds)', data=df, palette='coolwarm')
        plt.title('Response Latency Distribution by Complexity')
        plt.ylabel('Latency (Seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4.5_Latency_by_Complexity.png'), dpi=300)
        plt.close()

        # Visual 6: Performance by Query Type
        print("Generating Figure 4.6: Accuracy by Query Type...")
        plt.figure(figsize=(10, 6))
        stats_type = df.groupby('Query Type').mean(numeric_only=True).reset_index().sort_values('Accuracy Score', ascending=False)
        sns.barplot(x='Accuracy Score', y='Query Type', data=stats_type, palette='magma')
        plt.xlim(0, 1.05)
        plt.title('System Accuracy by Query Type')
        plt.xlabel('Average Accuracy Score')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4.6_Accuracy_by_QueryType.png'), dpi=300)
        plt.close()
        
        # Visual 7: Latency vs. Accuracy Scatter
        print("Generating Figure 4.7: Latency vs Accuracy...")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Response Latency (Seconds)', y='Accuracy Score', hue='Complexity Level', palette='deep', alpha=0.7)
        plt.title('Response Latency vs. Accuracy Score')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Fig4.7_Latency_vs_Accuracy.png'), dpi=300)
        plt.close()

        print("Evaluation Visuals Complete.")

    except Exception as e:
        print(f"Error generating evaluation visuals: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_eda_visuals()
    generate_evaluation_visuals()
    print(f"\nAll visuals saved to: {OUTPUT_DIR}")
