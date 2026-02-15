import unittest
from chat import DietCheatEngine

# --- CONFIG ---
# This script BYPASSES the 'Franco' node entirely to isolate Math Logic resolution.
# It feeds Arabic text DIRECTLY into '_node_math'.

class TestMathStress(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n=== SETUP: Initializing Engine (Loading Embeddings...) ===")
        cls.engine = DietCheatEngine()
        print(f"Food Index Loaded: {len(cls.engine.food_index['all_items'])} items.")

    def run_math_node_direct(self, query):
        state = {
            "req_id": "test_math",
            "raw_query": query,
            "normalized_query": query, # BYPASS FRANCO
            "intent": "MATH_SUBSTITUTION",
            "sources": []
        }
        class DummyTrace:
            def log(self, *args, **kwargs): pass
            def write_jsonl(self): pass
        trace = DummyTrace()

        self.engine._node_math(state, trace)
        return state

    def test_math_stress_20_arabic(self):
        print("\n=== TEST: 20 Arabic Math Queries (Stress Test) ===")

        # Format: (Query, Expected Value, Tolerance)
        # Includes synonyms (Turkey/Rumi, Frakh/Dajaj, Ruz/Arz) to test Fuzzy/Vector.
        examples = [
            # 1. Chicken variants
            ("بدل 100 جرام فراخ مشويه ب لحمه مسلوقه", 100, 40),       # "Frakh" (Vector/Fuzzy should catch this)
            ("حول 100 جرام دجاج ب سمك", 100, 30), # Exact "Sudur Dajaj"

            # 2. Rice/Carb variants
            ("عايز ابدل 50 جرام ارز ب معكرونة", 50, 20),   # "Ruz" vs "Arz"
            ("بدل 60 جرام أرز ب بطاطسس", 260, 50),        # Exact "Arz"
            ("ابدل 200 جرام بطاطس ب عيش", 70, 25),        # Potato -> Bread

            # 3. Fish variants
            ("غير 100 جرام تونة ب فيليه", 120, 30),  # Tuna -> Fillet
            ("بدل 120 جرام سالامون ب جمبري", 100, 30),     # Salmon -> Shrimp

            # 4. Dairy/Breakfast
            ("بدل 30 جرام شوفان ب كورن فليكس", 30, 10),  # Oats -> Cornflakes
            ("بدل 200 جرام جبن قريش ب بيض", 250, 50),    # Cottage Cheese -> Eggs
            ("بدل 240 جرام لبن كامل الدسم ب زبادي", 240, 30),

            # 5. Fruits
            ("بدل 100 جرام تفاح ب موز", 60, 20),         # Apple -> Banana
            ("بدل 100 جرام جوافة ب برتقال", 100, 20),    # Guava -> Orange

            # 6. Fats
            ("بدال 15 جرام زيت زيتون ب زبدة", 15, 5),     # Olive Oil -> Butter
            ("بدل 20 جرام سوداني ب لوز", 20, 5),         # Peanuts -> Almonds

            # 7. Meats (Harder)
            ("استبدل 120 جرام تركي ب صدور دجاج", 120, 20),  # "Turkey" (English in Arabic) vs "Deek Rumi"
            ("حول 100 جرام لحم ب كبده", 100, 30),        # Meat -> Liver

            # 8. Vegetables (Low cal)
            ("غير 100 جرام خيار ب طماطم", 100, 20),      # Cucumber -> Tomato
            ("بدل 50 جرام فلفل ب جزر ", 50, 20),          # Pepper -> Carrot

            # 9. Complex/Messy
            ("عايز 50 جرام شوفان بدل  بليلة", 50, 10) # Oats <-> Balila
        ]

        success_count = 0
        for i, (q, expected_val, tolerance) in enumerate(examples):
            print(f"\n[{i+1}] Query: {q}")

            state = self.run_math_node_direct(q)

            error = state.get("math_error")
            res = state.get("math_result", {})

            if error:
                print(f" -> ❌ FAIL (Error): {error}")
            else:
                calc_val = res.get("res", 0)
                src = res.get("src", "?")
                tgt = res.get("tgt", "?")
                method = res.get("debug_methods", "unknown")
                print(f" -> Result: {calc_val:.1f}g ({src} -> {tgt}) [Method: {method}]")

                # Validation Logic
                diff = abs(calc_val - expected_val)
                if diff <= tolerance:
                    print(f" -> ✅ PASSED")
                    success_count += 1
                else:
                    print(f" -> ⚠️ VALUE MISMATCH (Exp: {expected_val}, Got: {calc_val:.1f})")

        print(f"\nMath Success Rate: {success_count}/{len(examples)}")
        self.assertGreaterEqual(success_count, 15, "Math logic success rate too low (<75%)")

if __name__ == "__main__":
    unittest.main()