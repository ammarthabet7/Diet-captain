import time
import json
import os
from datetime import datetime
from colorama import Fore, Style, init

try:
    from chat import DietCheatEngine, TraceLogger
except ImportError:
    print(Fore.RED + "CRITICAL: Could not import DietCheatEngine from chat.py" + Style.RESET_ALL)
    exit(1)

init(autoreset=True)

# CORRECT TEST DATA
TEST_DATA = [
    # English
    ("What is RIR in training?", ["reserve", "failure", "reps"], "EN_PURE"),
    ("How to calculate TEF?", ["thermic", "calories", "digestion"], "EN_PURE"),
    ("Is creatine safe?", ["kidney", "water", "safe", "research"], "EN_PURE"),
    # Mixed
    ("يعني ايه Drop Set؟", ["وزن", "تخفيف", "مجموعة", "فشل"], "MIXED_AR_EN"),
    ("ازاي اعمل Progressive Overload صح؟", ["زيادة", "حمل", "وزن", "تكرار"], "MIXED_AR_EN"),
    ("ايه هو ال Super Set؟", ["تمرينين", "راحة", "عضلة"], "MIXED_AR_EN"),
    ("ايه فايدة ال Deload week؟", ["راحة", "استشفاء", "أوزان"], "MIXED_AR_EN"),
    ("يعني ايه RIR؟", ["عدات", "احتياطي", "فشل"], "MIXED_AR_EN"),
    ("الفرق بين Simple Carbs و Complex Carbs", ["بسيط", "معقد", "شوفان", "سكر"], "MIXED_AR_EN"),
    ("ازاي احسب ال Macros بتاعتي؟", ["بروتين", "دهون", "كاربو", "سعرات"], "MIXED_AR_EN"),
    ("هل ال Whey Protein مضر للكلى؟", ["آمن", "خرافة", "بروتين", "طبيعي"], "MIXED_AR_EN"),
    ("امتى اخد ال Creatine؟", ["وقت", "تمرين", "أي وقت", "تراكمي"], "MIXED_AR_EN"),
    ("ايه هو ال Caloric Deficit؟", ["عجز", "سعرات", "حرق", "دهون"], "MIXED_AR_EN"),
    ("هل ال Cardio بيحرق عضل؟", ["خرافة", "شدة", "تغذية"], "MIXED_AR_EN"),
    ("افضل تمارين لل Chest", ["نش", "تجميع", "ضغط", "صدر"], "MIXED_AR_EN"),
    ("اعراض ال Overtraining ايه؟", ["أرق", "إرهاق", "إصابات", "نفسية"], "MIXED_AR_EN"),
    ("ايه فايدة ال Caffeine قبل التمرين؟", ["طاقة", "تركيز", "أداء"], "MIXED_AR_EN"),
    ("ازاي اعمل Bulk نظيف؟", ["فائض", "سعرات", "دهون", "عضل"], "MIXED_AR_EN")
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>RAG Mixed Test</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background-color: #f4f4f9; padding: 20px; text-align: right; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        .summary {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; display: flex; justify-content: space-around; }}
        table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 15px; text-align: right; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #34495e; color: #fff; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #c0392b; font-weight: bold; }}
        .chunk-box {{ background: #ecf0f1; padding: 8px; margin: 4px 0; border-right: 4px solid #bdc3c7; font-size: 0.85em; text-align: left; direction: ltr; }}
        .answer-box {{ background: #e8f6f3; padding: 10px; border-right: 4px solid #1abc9c; margin-top: 10px; }}
        .details-row {{ display: none; background-color: #fcfcfc; }}
        .details-btn {{ cursor: pointer; color: #2980b9; text-decoration: underline; }}
    </style>
    <script>
        function toggleDetails(id) {{
            var row = document.getElementById('details-' + id);
            row.style.display = row.style.display === 'table-row' ? 'none' : 'table-row';
        }}
    </script>
</head>
<body>
    <h1>RAG Stress Test (Mixed Arabic + English)</h1>
    <div class="summary">
        <div>Total: <b>{total}</b></div>
        <div style="color: #27ae60">Passed: <b>{passed}</b></div>
        <div style="color: #c0392b">Failed: <b>{failed}</b></div>
    </div>
    <table>
        <thead><tr><th>#</th><th>Query</th><th>Type</th><th>Score</th><th>Status</th><th>Action</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
</body>
</html>
"""

def run_test():
    print(Fore.CYAN + "Initializing Mixed Stress Test...")
    engine = DietCheatEngine()
    print(Fore.GREEN + "Engine Loaded.\n")
    
    results_html = []; passed_count = 0
    for i, (query, keywords, test_type) in enumerate(TEST_DATA, 1):
        print(f"Running Test {i}/{len(TEST_DATA)}: {query} ...", end="\r")
        req_id = f"mixed_{i}"
        trace = TraceLogger(req_id)
        state = {"req_id": req_id, "raw_query": query, "normalized_query": query, "intent": "GENERAL_QA", "history": [], "chunks": [], "sources": []}
        
        try:
            engine._node_rag(state, trace)
            engine._node_writer(state, trace)
        except Exception as e:
            state["final_answer"] = f"ERROR: {e}"
        
        final_ans = state.get("final_answer", "")
        found = [k for k in keywords if k.lower() in final_ans.lower()]
        is_pass = len(found) > 0
        if is_pass: passed_count += 1
        
        chunks = state.get("chunks", [])
        top_score = chunks[0].get("score", 0.0) if chunks else 0.0
        chunks_html = "".join([f"<div class='chunk-box'><b>[{c.get('score',0):.3f}] {c['meta']['source']}</b><br>{c['text'][:180]}...</div>" for c in chunks])
        
        row = f"""
        <tr><td>{i}</td><td dir="auto">{query}</td><td>{test_type}</td><td>{top_score:.4f}</td>
        <td class="{'pass' if is_pass else 'fail'}">{'PASS' if is_pass else 'FAIL'}</td>
        <td><span class="details-btn" onclick="toggleDetails({i})">Details</span></td></tr>
        <tr id="details-{i}" class="details-row"><td colspan="6">
        <strong>Expected:</strong> {', '.join(keywords)}<br><strong>Found:</strong> {', '.join(found)}<br>
        <div class="answer-box"><strong>Answer:</strong><br>{final_ans}</div>
        <strong>Context:</strong>{chunks_html}</td></tr>
        """
        results_html.append(row)

    with open("stress_test_mixed.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.format(total=len(TEST_DATA), passed=passed_count, failed=len(TEST_DATA)-passed_count, rows="\n".join(results_html)))
    print(Fore.GREEN + f"\n\n✅ DONE! Report: {os.path.abspath('stress_test_mixed.html')}")

if __name__ == "__main__":
    run_test()
