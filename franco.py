import time
import os
import json
from datetime import datetime
from colorama import Fore, Style, init

try:
    from chat import DietCheatEngine, TraceLogger
except ImportError:
    print(Fore.RED + "CRITICAL: Could not import DietCheatEngine from chat.py" + Style.RESET_ALL)
    exit(1)

init(autoreset=True)

TEST_CASES = [
    "ezayak yasta عامل ايه",
    "el creatine by3mel water retention?",
    "3ayez a5es 5 kgs f shahr",
    "eh afdal tamren lel chest?",
    "momken badil lel shofan?",
    "hal el whey protein modr?",
    "ana ba7ess b fatigue b3d el tamren",
    "ezay a3mel progressive overload?",
    "kam gram protein f el 100g ferakh?",
    "el macros bta3ty 2000 cal",
    "3ayez atmaran 4 days split",
    "eh el far2 ben isolation w compound?",
    "ana beginner w 3ayez abda2 gym",
    "el nom mohem lel recovery?",
    "hal el coffee btwa2af el 7ar2?"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>Franco Test</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background-color: #f8f9fa; padding: 20px; text-align: right; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 15px; border-bottom: 1px solid #eee; }}
        th {{ background-color: #34495e; color: #fff; }}
        .raw-text {{ font-family: 'Courier New', monospace; direction: ltr; text-align: left; color: #e67e22; font-weight: bold; }}
        .trans-text {{ font-size: 1.2em; color: #27ae60; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Franco-Arabic Translation Test (Qwen 2.5)</h1>
    <div style="text-align: center; margin-bottom: 20px;">Date: {date}</div>
    <table>
        <thead><tr><th>#</th><th style="text-align: left;">Raw Input (Franco)</th><th>Translated Output (Arabic)</th><th>Latency</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
</body>
</html>
"""

def run_franco_test():
    print(Fore.CYAN + "Initializing Franco Node Test...")
    engine = DietCheatEngine()
    print(Fore.GREEN + "Engine Loaded. Starting Tests...\n")
    results_html = []
    
    for i, raw_query in enumerate(TEST_CASES, 1):
        print(f"Translating {i}/{len(TEST_CASES)}: {raw_query} ...", end="\r")
        req_id = f"franco_{i}"
        trace = TraceLogger(req_id)
        state = {"req_id": req_id, "raw_query": raw_query, "normalized_query": "", "history": []}
        start_t = time.perf_counter()
        
        try:
            engine._node_franco(state, trace)
            output = state["normalized_query"]
        except Exception as e:
            output = f"ERROR: {e}"
        
        duration = time.perf_counter() - start_t
        row = f"<tr><td>{i}</td><td class='raw-text'>{raw_query}</td><td class='trans-text'>{output}</td><td>{duration:.2f}s</td></tr>"
        results_html.append(row)

    with open("franco_test_report.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rows="\n".join(results_html)))
    print(Fore.GREEN + f"\n\n✅ DONE! Report: {os.path.abspath('franco_test_report.html')}")

if __name__ == "__main__":
    run_franco_test()
