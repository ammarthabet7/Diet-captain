import json
import os

path = "ready_data/food_exchange_clean.json"
if not os.path.exists(path):
    print(f"Error: {path} not found.")
    exit()

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# EXTENDED MAPPING based on stress test failures
manual_aliases = {
    # Existing ones
    "شوفان": ["shofan", "oats", "oatmeal"],
    "أرز": ["roz", "rice", "white rice", "ruz", "ارز", "رز"], 
    "لحم صدور": ["chicken", "frakh", "chicken breast", "فراخ", "دجاج", "صدور دجاج"],
    "فينو": ["fino", "3esh fino", "bread", "عيش"],
    "بطاطس": ["batates", "potato", "potatoes"],
    "بيض كامل": ["egg", "eggs", "whole egg", "bed", "بيض"],
    "موز": ["moz", "banana"],
    
    # NEW FIXES (The Critical Ones)
    "مكرونه": ["مكرونة", "makarona", "pasta", "spaghetti", "macaroni"], 
    "زبادي كامل الدسم": ["زبادي", "zabadi", "yogurt", "youghurt", "zabadry"],
    "فصوص رومي": ["تركي", "turkey", "roomi", "romi", "deek roomi"],
    "لبن كامل الدسم": ["لبن", "milk", "laban", "halib"],
    "جبن قريش": ["جبنة قريش", "kareesh", "cottage cheese", "quraish cheese"],
    "سمك فيليه": ["فيليه", "fillet", "fish fillet"],
    "تونه": ["تونا", "tuna", "canned tuna"],
    "فول سوداني": ["سوداني", "peanuts", "sodani"],
    "لوز": ["almond", "almonds", "loz"],
    "جمبري": ["shrimp", "gambary"],
    "سمك سالمون": ["salmon", "salamon"],
    "بليلة": ["بليلة", "balila"]
}

count = 0
for cat in data:
    for item in cat.get("items", []):
        name = item.get("item_ar", "").strip()
        
        # Match keys against item name
        matched_keys = [k for k in manual_aliases.keys() if k == name or (name and k in name)]
        
        for k in matched_keys:
            v_list = manual_aliases[k]
            # Initialize aliases if missing
            if "aliases" not in item: item["aliases"] = []
            
            current = set(item.get("aliases", []))
            for v in v_list:
                if v not in current:
                    item["aliases"].append(v)
                    count += 1
            
            # Clean up
            item["aliases"] = list(set(item["aliases"]))

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Force updated {count} aliases in {path}.")
