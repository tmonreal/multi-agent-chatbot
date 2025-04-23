from llm_engine import load_multi_agents, classify_query_target, run_combined_rag

# Load the agents and the LLM used for classification
qa_chains, groq_model = load_multi_agents()

# These queries are designed to test the classification and retrieval capabilities of the system
test_queries = [
    "Where did each of them study?",
    "What programming languages does she know?",
    "Where did Jorge go to school?",
    "Tell me about her education.",
    "Compare their work experience.",
    "What are their top skills?"
]

for query in test_queries:
    print(f"\nQuery: {query}❓")
    
    target = classify_query_target(groq_model, query)
    print(f"🔍 LLM classified the query as: {target}")

    # Dispatch to the correct agent(s)
    if target == "both":
        final_answer = run_combined_rag(query, qa_chains, groq_model, debug=True)
    elif target == "jorge":
        final_answer = qa_chains["jorge"].invoke(query)["result"]
    else:
        final_answer = qa_chains["trinidad"].invoke(query)["result"]

    print("🧠 Final Answer:\n", final_answer)
