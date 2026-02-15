import csv
import json
import time
import os
import re
from typing import Dict, Any, List

#from colorama import Fore, Style, init
# from colorama import Fore, Style, init   <-- COMMENT THIS OUT
# init(autoreset=True)                     <-- COMMENT THIS OUT

# Add these dummy classes so you don't have to delete Fore.RED everywhere
class Fore:
    RED = ""
    GREEN = ""
    CYAN = ""
    YELLOW = ""
    
class Style:
    RESET_ALL = ""

# --- IMPORT ENGINE ---
try:
    import chat  # run_turn()
    print("USING chat.py from:", chat.__file__)  # verify which file is running
except ImportError:
    print(Fore.RED + "CRITICAL: Could not import chat.py" + Style.RESET_ALL)
    exit(1)

#init(autoreset=True)

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

GROUND_TRUTH = {

    # MATH SUBSTITUTION (2 Examples)
    "بديل 100 جرام فراخ بـ لحمة": {"intent": "MATH_SUBSTITUTION", "expect_val": 100, "tolerance": 20},
    "بديل 60 جرام رز بـ مكرونة": {"intent": "MATH_SUBSTITUTION", "expect_val": 65, "tolerance": 10}
}


QUERIES = list(GROUND_TRUTH.keys())

def get_latest_trace():
    """Reads the JSONL log file and finds the latest entry"""
    log_dir = "logs"
    if not os.path.exists(log_dir): return {}
    files = sorted([f for f in os.listdir(log_dir) if f.startswith("trace_")])
    if not files: return {}
    
    # Read the very last line of the newest log file
    try:
        with open(os.path.join(log_dir, files[-1]), "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines: return json.loads(lines[-1])
    except: pass
    return {}

def extract_number_from_answer(text):
    # SAFETY CHECK: If text is None, return 0.0 immediately
    if not text:
        return 0.0
        
    # Extracts the first number found (handling float points)
    # Looks for pattern *123* or just 123
    match = re.search(r"(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return 0.0


def run_tests():
    results = []
    print(Fore.CYAN + f"\n🚀 Starting Full Pipeline Test on {len(QUERIES)} queries...\n")
    
    metrics = {"total": 0, "intent_correct": 0, "math_correct": 0, "math_total": 0, "failures": 0}

    for i, q in enumerate(QUERIES):
        print(f"[{i+1}/{len(QUERIES)}] {q[:40]}... ", end="", flush=True)
        t0 = time.time()
        
        try:
            # 1. Run Engine
            response = chat.run_turn(q, [])
            time.sleep(0.2) # Allow log flush
            
            # 2. Get Trace Info
            trace_data = get_latest_trace()
            summary = trace_data.get("summary", {})
            # --- DEBUG: PRINT FULL TRACE ---
            print("\n" + "-"*30 + " DEBUG TRACE " + "-"*30)
            print(f"RAW TRACE DATA: {json.dumps(trace_data, indent=2, ensure_ascii=False)}")
            print("-" * 75 + "\n")
            # -------------------------------

            # Extract Deep Metrics
            actual_intent = summary.get("intent", "UNKNOWN")
            translated_q = summary.get("translated_query", "-")
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            
            # 3. Determine Expectations
            truth = GROUND_TRUTH[q]
            expected_intent = truth if isinstance(truth, str) else truth["intent"]
            
            # 4. Check Intent
            intent_match = (actual_intent == expected_intent)
            metrics["total"] += 1
            if intent_match: metrics["intent_correct"] += 1

            # 5. Check Math
            math_status = "N/A"
            if isinstance(truth, dict) and "expect_val" in truth:
                metrics["math_total"] += 1
                val = extract_number_from_answer(answer)
                target = truth["expect_val"]
                tol = truth["tolerance"]
                
                if abs(val - target) <= tol:
                    metrics["math_correct"] += 1
                    math_status = f"✅ Pass ({val}g)"
                else:
                    math_status = f"❌ Fail (Got {val}g, Exp {target}g)"
            
            # Console Log
            if intent_match and ("❌" not in math_status):
                print(Fore.GREEN + "✅ PASS")
            else:
                print(Fore.RED + f"❌ FAIL (Intent: {actual_intent}, Math: {math_status})")
            
            # 6. Build Result Row
            results.append({
                "query": q,
                "translated": translated_q,
                "answer": answer,
                "actual_intent": actual_intent,
                "expected_intent": expected_intent,
                "math_status": math_status,
                "sources": sources,
                "latency": (time.time() - t0)
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

            
    # --- Summary ---
    print("\n" + "="*40)
    intent_acc = (metrics["intent_correct"] / metrics["total"]) * 100 if metrics["total"] else 0
    math_acc = (metrics["math_correct"] / metrics["math_total"]) * 100 if metrics["math_total"] else 0
    
    print(f"Total Queries:   {metrics['total']}")
    print(f"Intent Accuracy: {intent_acc:.1f}%")
    if metrics["math_total"] > 0:
        print(f"Math Accuracy:   {math_acc:.1f}%")
    print("="*40)
    
    save_html_report(results, metrics)
    print(Fore.GREEN + f"\n📄 Report saved to {os.path.abspath(os.path.join(OUT_DIR, 'report.html'))}")

def save_html_report(rows, metrics):
    path = os.path.join(OUT_DIR, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("""
        <!DOCTYPE html>
        <html lang="ar">
        <head>
            <meta charset="UTF-8">
            <title>Pipeline Test Report</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px; }
                h2 { color: #2c3e50; text-align: center; }
                .summary-box { display: flex; justify-content: space-around; background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px; }
                .metric { text-align: center; }
                .metric-val { font-size: 1.5em; font-weight: bold; color: #2980b9; }
                
                table { border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
                th, td { padding: 12px 15px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; }
                th { background-color: #34495e; color: #fff; font-weight: 600; }
                
                .pass { border-left: 5px solid #27ae60; }
                .fail { border-left: 5px solid #c0392b; background-color: #fff5f5; }
                
                .chunk { background: #f8f9fa; padding: 8px; margin: 4px 0; border-left: 3px solid #bdc3c7; font-size: 0.85em; color: #555; }
                .source-tag { font-size: 0.75em; color: #7f8c8d; font-weight: bold; margin-bottom: 2px; }
                
                .answer-box { background: #e8f8f5; padding: 10px; border-radius: 6px; font-size: 0.95em; white-space: pre-wrap; direction: rtl; }
                .trans-box { font-size: 0.85em; color: #e67e22; font-weight: bold; margin-top: 5px; }
                
                .badge { padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
                .badge-intent { background: #e0f7fa; color: #006064; }
                .badge-math { background: #f3e5f5; color: #4a148c; }
            </style>
        </head>
        <body>
        """)
        
        f.write(f"<h2>🚀 Pipeline Integration Test</h2>")
        
        # Summary
        i_acc = (metrics['intent_correct'] / metrics['total']) * 100 if metrics['total'] else 0
        m_acc = (metrics['math_correct'] / metrics['math_total']) * 100 if metrics['math_total'] else 0
        
        f.write(f"""
        <div class="summary-box">
            <div class="metric">Total Queries<div class="metric-val">{metrics['total']}</div></div>
            <div class="metric">Intent Accuracy<div class="metric-val" style="color: {'#27ae60' if i_acc > 90 else '#e67e22'}">{i_acc:.1f}%</div></div>
            <div class="metric">Math Accuracy<div class="metric-val" style="color: {'#27ae60' if m_acc > 80 else '#c0392b'}">{m_acc:.1f}%</div></div>
        </div>
        """)
        
        f.write("<table><thead><tr>")
        f.write("<th width='20%'>Query & Translation</th>")
        f.write("<th width='15%'>Intent Logic</th>")
        f.write("<th width='30%'>RAG Retrieval</th>")
        f.write("<th width='35%'>Final Answer (WhatsApp)</th>")
        f.write("</tr></thead><tbody>")
        
        for r in rows:
            is_intent_pass = r['expected_intent'] == r['actual_intent']
            is_math_pass = "❌" not in r['math_status']
            row_class = "pass" if (is_intent_pass and is_math_pass) else "fail"
            
            f.write(f"<tr class='{row_class}'>")
            
            # Col 1: Query
            f.write(f"<td><div>{r['query']}</div><div class='trans-box'>→ {r['translated']}</div></td>")
            
            # Col 2: Intent
            status_icon = "✅" if is_intent_pass else "❌"
            f.write(f"<td><span class='badge badge-intent'>{r['actual_intent']}</span><br><br>{status_icon} Exp: {r['expected_intent']}<br><br><span class='badge badge-math'>{r['math_status']}</span></td>")
            
            # Col 3: Retrieval
            f.write("<td>")
            if r['sources']:
                for s in r['sources']:
                    f.write(f"<div class='chunk'><div class='source-tag'>{s['source']}</div>{s['snippet']}...</div>")
            else:
                f.write("<span style='color:#ccc'>No Context</span>")
            f.write("</td>")
            
            # Col 4: Answer
            f.write(f"<td><div class='answer-box'>{r['answer']}</div></td>")
            
            f.write("</tr>")
            
        f.write("</tbody></table></body></html>")

if __name__ == "__main__":
    run_tests()
