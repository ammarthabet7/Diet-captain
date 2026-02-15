import time
from retriever import DietRetriever

# Load Questions
questions = [
    "What is RIR?", "شرح ال dropset", "ايه هو ال deload", "How to calculate 1RM?", 
    "الفرق بين aerobic و anaerobic", "EMOM definition", "Best cardio for fat loss?", 
    "ازاي اسخن قبل التمرين", "Is FST-7 good?", "meaning of failure in gym",
    "بديل العيش الفينو", "I don't like eggs, what protein can I eat?", "بديل الرز البسمتي", 
    "كم سعرة في 100 جرام فراخ؟", "Substitute for 50g Oats?", "بديل الزبدة في الدهون", 
    "I am allergic to milk, alternatives?", "سعرات الموز", "بديل البطاطس", "Can I swap chicken for fish?",
    "هل الكرياتين مضر؟", "How to sleep better for recovery?", "يعني ايه سعرات ثبات؟", 
    "I have cravings for sugar, what to do?", "هل الاكل بليل بيتخن؟", "benefits of water", 
    "how to track macros", "Whey protein necessity?", "الفرق بين الميزان قبل وبعد الطبخ", "How to start a diet?"
]

def main():
    print("Loading Retriever (BGE-M3)... This will take a moment.")
    retriever = DietRetriever()
    
    results_log = []
    
    print(f"Starting Evaluation on {len(questions)} questions...")
    start_time = time.time()
    
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Searching: {q}")
        
        # Search (Hybrid + Rerank)
        chunks = retriever.search(q, k=3)
        
        # Format Output
        log_entry = f"\n{'='*50}\nQ{i+1}: {q}\n{'='*50}\n"
        for j, c in enumerate(chunks):
            # Clean newlines for display
            preview = c['text'].replace('\n', ' ')[:150] 
            log_entry += f"#{j+1} (Score: {c['score']:.4f}) [{c.get('source', 'Unknown')}]\n   {preview}...\n"
            
        results_log.append(log_entry)

    total_time = time.time() - start_time
    print(f"\nFinished in {total_time:.2f} seconds ({total_time/len(questions):.2f}s per query).")
    
    # Save to file
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("".join(results_log))
    
    print("Results saved to 'evaluation_results.txt'. Open it to verify!")

if __name__ == "__main__":
    main()
