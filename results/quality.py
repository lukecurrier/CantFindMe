import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import textstat
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import matplotlib.pyplot as plt
import numpy as np

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

def compute_perplexity(text):
    """Calculates perplexity using GPT-2."""
    inputs = gpt2_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,  
        max_length=512
    )
    with torch.no_grad():
        loss = gpt2_model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


def compute_ttr(text):
    """Calculates Type-Token Ratio (TTR)."""
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0


def compute_self_bleu(outputs):
    """Calculates Self-BLEU for a list of outputs."""
    scores = []
    for i, output in enumerate(outputs):
        others = outputs[:i] + outputs[i + 1:]
        scores.append(sentence_bleu([o.split() for o in others], output.split()))
    return sum(scores) / len(scores) if scores else 0


def compute_readability(text):
    """Calculates readability metrics."""
    return {
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "automated_readability": textstat.automated_readability_index(text)
    }


def flatten_nested_list(nested_list):
    """Flattens a nested list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]


def evaluate_quality(original_outputs):
    """Evaluates various metrics for original_outputs."""

    results = {
        "perplexity": [],
        "text_diversity": [],
        "readability": [],
        #"self_bleu": None
    }

    results["perplexity"] = [compute_perplexity(output) for output in original_outputs]
    results["text_diversity"] = [compute_ttr(output) for output in original_outputs]
    results["readability"] = [compute_readability(output) for output in original_outputs]
    #results["self_bleu"] = compute_self_bleu(original_outputs)

    return results

def plot_metrics(model_name: str, results: dict, index: int):
    """Plots the metrics for a given model."""
    perplexity = results['perplexity']
    ttr = results['text_diversity']
    readability = results['readability']

    # Extract readability metrics
    flesch_kincaid = [r['flesch_kincaid'] for r in readability]
    gunning_fog = [r['gunning_fog'] for r in readability]
    automated_readability = [r['automated_readability'] for r in readability]

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Quality Metrics for {model_name}", fontsize=16)

    # 1. Perplexity - with color gradient based on value
    x = np.arange(len(perplexity))
    y = np.array(perplexity)

    norm = plt.Normalize(vmin=1, vmax=100)
    my_cmap = plt.get_cmap("turbo")
    axs[0, 0].bar(x, y, color=my_cmap(norm(y)))

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=my_cmap, norm=norm), ax=axs[0, 0])
    cbar.set_label('Perplexity Scale')

    axs[0, 0].set_title("Perplexity per Output")
    axs[0, 0].set_xlabel("Output Index")
    axs[0, 0].set_ylabel("Perplexity")
    axs[0, 0].set_ylim([0.0, 100.0])

    # 2. Text Diversity (TTR)
    axs[0, 1].plot(range(len(ttr)), ttr, marker='o', color='green')
    axs[0, 1].set_title("Text Diversity (TTR)")
    axs[0, 1].set_xlabel("Output Index")
    axs[0, 1].set_ylabel("TTR")
    axs[0, 1].set_ylim(0, 1)

    # 3. Readability
    indices = range(len(flesch_kincaid))
    width = 0.25
    axs[1, 0].bar(indices, flesch_kincaid, width, label="Flesch-Kincaid", color='blue')
    axs[1, 0].bar([i + width for i in indices], gunning_fog, width, label="Gunning Fog", color='orange')
    axs[1, 0].bar([i + 2 * width for i in indices], automated_readability, width, label="Automated Readability", color='purple')
    axs[1, 0].set_title("Readability Metrics")
    axs[1, 0].set_xlabel("Output Index")
    axs[1, 0].set_ylabel("Readability Score")
    axs[1, 0].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    plt.savefig(f"model_{index}_metrics.png")
    plt.show()

def plot_text_diversity(models, ttr_values):
    """
    Plots the Text Diversity (TTR) line plot for multiple models.
    
    models: List of model names.
    ttr_values: List of TTR values for each model. Each model gets a list of TTR values.
    """

    # Create a color palette for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))  # tab10 colormap with enough colors

    plt.figure(figsize=(10, 6))

    for i, (model, ttr) in enumerate(zip(models, ttr_values)):
        plt.plot(range(1, len(ttr) + 1), ttr, marker='o', label=model, color=colors[i], linewidth=2, linestyle='-', markersize=6)

    plt.title("Text Diversity (TTR) per Output")
    plt.xlabel("Output Index")
    plt.ylabel("TTR")
    plt.legend(title="Models", loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    with open("results/eval_output12_1_2204.json", "r") as file:
        output = json.load(file)
    outputs = output["responses"]

    model_names = ["Llama-70B", "Llama-8B", "Gemma-9B", "Mistral-7B", "Qwen-7B"]

    results_list = []
    for i, model_outputs in enumerate(outputs):
        model_name = model_names[i]
        print(f"\nEvaluating Model: {model_name}")
        results = evaluate_quality(model_outputs)
        results_list.append(results)
    
    """ttr_values = []
    for result in results_list:
        if "text_diversity" in result:
            ttr_values.append(result["text_diversity"])
        else:
            print(f"Warning: 'text_diversity' key not found in result: {result}")
            ttr_values.append([])  # Append an empty list if the key is missing

    plot_text_diversity(model_names, ttr_values)"""

    plot_metrics(model_names[0], results_list[0], 0)
    plot_metrics(model_names[4], results_list[4], 4)
