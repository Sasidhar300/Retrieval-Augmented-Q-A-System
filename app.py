from flask import Flask, request, jsonify
from model import embedder, qa_pipeline, summary_pipeline, index, documents

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("question", "")

    # Generate embedding for the query
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().detach().numpy()

    # Search FAISS index for closest document
    D, I = index.search(query_embedding, k=1)
    closest_doc = documents[I[0][0]]
    print(f"Retrieved Document: {closest_doc}")

    # Summarize the document
    summary = summary_pipeline(closest_doc, max_length=300, min_length=50, do_sample=False)[0]['summary_text']
    print(f"Summary: {summary}")

    # Generate answer using QA pipeline
    context = f"Summary: {summary}\n\nQuestion: {query}\nAnswer:"
    response = qa_pipeline(context, max_new_tokens=100)
    print(f"Generated Response: {response[0]['generated_text']}")

    return jsonify({"answer": response[0]['generated_text']})

if __name__ == "__main__":
    app.run(debug=True)
