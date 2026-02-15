import dspy
import csv
from dspy.teleprompt import BootstrapFewShot

# 1. Setup DSPy (Correct V2.5+ Syntax)
lm = dspy.LM(
    model="ollama/hf.co/tensorblock/MBZUAI-Paris_Nile-Chat-4B-GGUF:Q4_K_M",
    api_base="http://localhost:11434",
    api_key="",
    max_tokens=512
)
dspy.settings.configure(lm=lm)

# 2. Define Signatures
class IntentClassifier(dspy.Signature):
    """Classify user query into intents: GENERAL_QA, CHIT_CHAT, MATH_SUBSTITUTION."""
    query = dspy.InputField()
    intent = dspy.OutputField(desc="One of: GENERAL_QA, CHIT_CHAT, MATH_SUBSTITUTION")

class DomainClassifier(dspy.Signature):
    """Classify query domain: nutrition_ebook, training_guide, both."""
    query = dspy.InputField()
    domain = dspy.OutputField(desc="One of: nutrition_ebook, training_guide, both")

# 3. Load Data & Fix Field Names
dataset = []
try:
    with open("dataset.txt", "r", encoding="utf-8") as f: # Use 'dataset.txt' if that's what you uploaded
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
except FileNotFoundError:
    print("Error: dataset.txt not found!")
    exit()

# 4. Create Examples with Correct Keys for Metrics
# DSPy's default metric checks 'example.answer' vs 'prediction.answer' (or matching field names)
# Since our output field is 'intent', we map 'intent' -> 'intent'
intent_examples = [
    dspy.Example(query=row["Query"], intent=row["Intent Label"]).with_inputs("query")
    for row in dataset
]

domain_examples = [
    dspy.Example(query=row["Query"], domain=row["RAG Domain Label"]).with_inputs("query")
    for row in dataset if row["Intent Label"] == "GENERAL_QA"
]

# 5. Define Custom Metric Wrappers
# We need to tell DSPy specifically which field to compare
def intent_metric(example, pred, trace=None):
    return example.intent == pred.intent

def domain_metric(example, pred, trace=None):
    return example.domain == pred.domain

# 6. Train Intent Router
print(f"\nTraining Intent Router ({len(intent_examples)} examples)...")
intent_teacher = dspy.ChainOfThought(IntentClassifier)
# Use our CUSTOM metric, not the default answer_exact_match
teleprompter = BootstrapFewShot(metric=intent_metric)
compiled_intent = teleprompter.compile(intent_teacher, trainset=intent_examples)

# 7. Train Domain Router
print(f"\nTraining Domain Router ({len(domain_examples)} examples)...")
domain_teacher = dspy.ChainOfThought(DomainClassifier)
teleprompter_dom = BootstrapFewShot(metric=domain_metric)
compiled_domain = teleprompter_dom.compile(domain_teacher, trainset=domain_examples)

# 8. Save
print("\n--- SAVING PROMPTS ---")
compiled_intent.save("intent_router_compiled.json")
print("Saved intent_router_compiled.json")

compiled_domain.save("domain_router_compiled.json")
print("Saved domain_router_compiled.json")
