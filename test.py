import os
import json
from model import embedder, qa_pipeline, summary_pipeline, index, documents
from difflib import SequenceMatcher

# Helper function for fuzzy matching
def is_match(generated, ground_truths):
    for gt in ground_truths:
        similarity = SequenceMatcher(None, generated.lower(), gt.lower()).ratio()
        if similarity > 0.7:  # Adjust threshold as needed
            return True
    return False

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
correct = 0
total = len(questions)
results = []

for i, question in enumerate(questions[:50]):  # Limit to 50 questions for testing
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

    # Match using semantic similarity
    if is_match(generated_answer, ground_truth_answers[i]):
        correct += 1
    else:
        results.append({
            "question": question,
            "expected": ground_truth_answers[i],
            "generated": generated_answer,
            "retrieved_document": closest_doc,
        })

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")

# Show errors for debugging
print("Examples of mismatches:")
for error in results[:10]:  # Show up to 10 errors
    print(f"Question: {error['question']}")
    print(f"Expected: {error['expected']}")
    print(f"Generated: {error['generated']}")
    print(f"Retrieved Document: {error['retrieved_document']}")
    print("---")
