# application.py

from flask import Flask, render_template, request, jsonify
from llm_engine import load_multi_agents, classify_query_target, run_combined_rag


app = Flask(__name__)

# Load the LLM + Retriever chain once on startup
qa_chains, groq_model = load_multi_agents()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_message = request.form.get("msg")
    if not user_message:
        return jsonify({"answer": "I didn't get your message."})
    
    try:
        target = classify_query_target(groq_model, user_message)

        if target == "both":
            final_answer = run_combined_rag(user_message, qa_chains, groq_model)
        elif target == "jorge":
            final_answer = qa_chains["jorge"].invoke({"query": user_message})["result"]
        else:
            final_answer = qa_chains["trinidad"].invoke({"query": user_message})["result"]

        return jsonify({"answer": final_answer})

    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)