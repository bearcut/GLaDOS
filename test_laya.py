import json
from pathlib import Path
import torch
from laya import Router

INPUT_FILE = "input.json"
OUTPUT_FILE = "output.json"

# Fallback evaluation schema
DEFAULT_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "Invoices, duplicate charges, payment failures, refunds.",
            "technical": "System errors, application crashes, connection outages.",
            "sales": "Pricing, quotes, enterprise licensing."
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "Evaluate the customer urgency level.",
        "criteria": [
            "low",
            "medium",
            "blocking"
        ]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user explicitly threaten to cancel or terminate service?"
    }
}

# 1. Hardware target setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on hardware target: {device}")

# 2. Model initialization
router = Router(device=device)

# 3. Read input JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 4. Predict (handles single item or batch list)
if isinstance(data, list):
    output_data = []
    for idx, entry in enumerate(data):
        state = entry.get("state", "")
        questions = entry.get("questions", DEFAULT_QUESTIONS)
        print(f"Evaluating item {idx + 1}/{len(data)}...")
        prediction = router.predict(state, questions)
        output_data.append({
            **entry,
            "prediction": prediction
        })
else:
    state = data.get("state", "")
    questions = data.get("questions", DEFAULT_QUESTIONS)
    print("Evaluating state...")
    prediction = router.predict(state, questions)
    output_data = {
        **data,
        "prediction": prediction
    }

# 5. Write predictions to output JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Finished. Results written to {OUTPUT_FILE}")