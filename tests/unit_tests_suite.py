import unittest
import json
import os
import sys
import time
from chat import DietCheatEngine

# --- CONFIG ---
# We force intents based on your provided GROUND_TRUTH.
# The focus is testing the LOGIC nodes (Math, RAG, ChitChat) and Franco normalization.

class TestDietEngineLogic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n=== SETUP: Initializing Engine ===")
        cls.engine = DietCheatEngine()
        if not cls.engine.food_index["all_items"]:
            print("WARNING: Food Index empty.")
        else:
            print(f"Food Index Loaded: {len(cls.engine.food_index['all_items'])} items.")

    def run_pipeline_with_forced_intent(self, query, forced_intent):
        req_id = "test_id"
        state = {
            "req_id": req_id,
            "raw_query": query,
            "normalized_query": query, 
            "history": [],
            "intent": forced_intent,
            "sources": []
        }
        class DummyTrace:
            def log(self, *args, **kwargs): pass
            def write_jsonl(self): pass
        trace = DummyTrace()

        # 1. Run Franco
        # We test if Franco normalization works by printing it.
        self.engine._node_franco(state, trace)
        norm_q = state["normalized_query"]

        # 2. Logic Branch (Skipping Router)
        if forced_intent == "MATH_SUBSTITUTION":
            self.engine._node_math(state, trace)
        elif forced_intent == "CHIT_CHAT":
            pass # Just falls through
        else:
            self.engine._node_rag(state, trace)

        # 3. Writer
        self.engine._node_writer(state, trace)

        return state

    # =========================================================================
    # 1. CHIT CHAT TESTS (8 Examples)
    # =========================================================================
    def test_chit_chat_group(self):
        print("\n\n=== TEST GROUP: CHIT CHAT (8 Examples) ===")
        examples = [
            "ezayak ya wa7sh"
            #"ezayak yasta", "3amel eh ya kbeer", "sabah el fol", "shokran ",
            #"nice to meet you", "anta meen?", "good night coach", "bye ya sa7by"
        ]

        for q in examples:
            print(f"\nQuery: {q}")
            state = self.run_pipeline_with_forced_intent(q, "CHIT_CHAT")
            ans = state.get("final_answer", "")
            norm = state.get("normalized_query", "")

            print(f" -> Franco Norm: {norm}")
            print(f" -> Answer: {ans[:100]}...") # Print first 100 chars

            self.assertTrue(len(ans) > 0, "Should have answer")
            self.assertEqual(len(state.get("sources", [])), 0, "No RAG for chitchat")

    # =========================================================================
    # 2. RAG QA TESTS (8 Examples)
    # =========================================================================
    def test_rag_group(self):
        print("\n\n=== TEST GROUP: RAG QA (8 Examples) ===")
        # Selecting diverse topics from your list
        examples = [
            "akol eh mmkn y3ly eltestosterone 3andy?",
            #"arya7 ad eh f elesboo3?",
            #"hal el fasting mofed le el 7ar2?",
            #"ana skinny fat a3mel bulk wala cut?",
            #"ya3ni eh drop sets?",
            #"eh a7san source carbs abl el gym",
            #"eh fwayed el zink w magnesium",
            #"ana hamouuut w akol haga mesakraa"

    
        ]

        passed_retrieval = 0
        for q in examples:
            print(f"\nQuery: {q}")
            state = self.run_pipeline_with_forced_intent(q, "GENERAL_QA")
            sources = state.get("sources", [])
            norm = state.get("normalized_query", "")
            ans = state.get("final_answer", "")

            print(f" -> Franco Norm: {norm}")
            print(f" -> Retrieved: {len(sources)} chunks")
            if sources:
                print(f" -> Top Source: {sources[0]['snippet'][:60]}...")
            print(f" -> Answer: {ans[:100]}...")

            if len(sources) > 0:
                passed_retrieval += 1

        print(f"\nRAG Retrieval Success Rate: {passed_retrieval}/{len(examples)}")
        # We expect most to find something, but allow 1-2 failures if topic is niche
        self.assertGreaterEqual(passed_retrieval, 6, "Should retrieve docs for at least 6/8 queries")


    # =========================================================================
    # 3. MATH SUBSTITUTION TESTS (8 Examples)
    # =========================================================================
    def test_math_group(self):
        print("\n\n=== TEST GROUP: MATH SUBSTITUTION (8 Examples) ===")
        # List of (Query, Expected Value, Tolerance)
        examples = [
            ("بدل 100 جرام فراخ ب لحمه", 100, 30),
            #("change 60g rice to pasta", 65, 20),
            ("بدل 15g زيت ب زبدة", 15, 5),
            #("law m3aya 200g batates a5od 3eish ad eh", 75, 25),
            ("بدل 100g تونة ب سمك", 120, 30),
            ("بدل 30g شوفان ب كورن فليكس", 30, 10),
            #("change 120g turkey to chicken", 120, 15),
            ("بدل 100g apple ب banana", 60, 20)
        ]

        success_count = 0
        for q, expected_val, tolerance in examples:
            print(f"\nQuery: {q}")
            state = self.run_pipeline_with_forced_intent(q, "MATH_SUBSTITUTION")

            error = state.get("math_error")
            res = state.get("math_result", {})
            norm = state.get("normalized_query", "")

            print(f" -> Franco Norm: {norm}")

            if error:
                print(f" -> MATH FAIL (Error): {error}")
            else:
                calc_val = res.get("res", 0)
                src = res.get("src", "?")
                tgt = res.get("tgt", "?")
                print(f" -> Result: {calc_val:.1f}g ({src} -> {tgt})")

                # Validation Logic
                diff = abs(calc_val - expected_val)
                if diff <= tolerance:
                    print(f" -> ✅ PASSED (Exp: {expected_val}, Got: {calc_val:.1f})")
                    success_count += 1
                else:
                    print(f" -> ❌ FAILED VALUE (Exp: {expected_val}, Got: {calc_val:.1f})")

        print(f"\nMath Success Rate: {success_count}/{len(examples)}")
        # We want to see if the model is smart enough now.
        # Fail the test if success is low (<50%)
        self.assertGreaterEqual(success_count, 4, "Math logic success rate too low (<50%)")

if __name__ == "__main__":
    unittest.main()