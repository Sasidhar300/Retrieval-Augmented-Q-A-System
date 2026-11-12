"""Diagnostic QA experiment, not a published SQuAD benchmark.

This loads heavy models and is separate from lightweight test_metrics.py.
"""

import os
import json

from model import embedder, qa_pipeline, summary_pipeline, index, documents
from evaluation_metrics import exact_match, token_f1

# Load SQuAD dev set for testing
def load_squad_questions(file_path):
    with open(file_path, "r") as f:
        squad_data = json.load(f)
    questions = []
    answers = []
    for topic in squad_data["data"]:
        for paragraph in topic["paragraphs"]:
            for qas in paragraph["qas"]:
                questions.append(qas["question"])
                answers.append([ans["text"] for ans in qas["answers"]])
    return questions, answers

# Load questions and answers
dev_file = os.path.join("documents", "dev-v1.1.json")
questions, ground_truth_answers = load_squad_questions(dev_file)

# Run the testing loop
exact_matches = 0
f1_sum = 0.0
# Only the evaluated subset belongs in the metric denominator.
total = min(50, len(questions))
results = []

for i, question in enumerate(questions[:total]):
    print(f"Testing Question {i+1}/{total}: {question}")
    
    # Retrieve the most relevant document
    query_embedding = embedder.encode([question], convert_to_tensor=True).cpu().detach().numpy()
    D, I = index.search(query_embedding, k=1)
    closest_doc = documents[I[0][0]]
    print(f"Retrieved Document: {closest_doc}")

    # Use the full document as context (skip summarization for better results)
    context = f"Context: {closest_doc}\n\nQuestion: {question}\nAnswer:"
    response = qa_pipeline(context, max_new_tokens=100)
    generated_answer = response[0]['generated_text']
    print(f"Generated Answer: {generated_answer}")
    print(f"Ground Truth Answers: {ground_truth_answers[i]}")

    # Deterministic text metrics, not "semantic similarity".
    em = exact_match(generated_answer, ground_truth_answers[i])
    f1 = token_f1(generated_answer, ground_truth_answers[i])
    exact_matches += int(em)
    f1_sum += f1
    if not em:
        results.append({
            "question": question,
            "expected": ground_truth_answers[i],
            "generated": generated_answer,
            "retrieved_document": closest_doc,
        })

# These scores are *diagnostic only*: the retrieval corpus is SQuAD train
# contexts while the questions are from dev, so this is not the standard
# SQuAD evaluation protocol. See README.md.
exact_match_percent = (exact_matches / total * 100) if total else 0.0
token_f1_percent = (f1_sum / total * 100) if total else 0.0
print(f"Evaluated questions: {total}")
print(f"Diagnostic exact match: {exact_match_percent:.2f}%")
print(f"Diagnostic token F1: {token_f1_percent:.2f}%")

# Show errors for debugging
print("Examples of mismatches:")
for error in results[:10]:  # Show up to 10 errors
    print(f"Question: {error['question']}")
    print(f"Expected: {error['expected']}")
    print(f"Generated: {error['generated']}")
    print(f"Retrieved Document: {error['retrieved_document']}")
    print("---")
