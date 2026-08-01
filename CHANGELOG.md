# Changelog

## 2026-07-31 — Prompt-based behaviour improvements

### Summary

The same LiveKit Retail agent, tools, models, database, and evaluation suite were used for both runs.

Only the system prompt changed.

Across 10 repeated evaluation runs:

- Target behaviours: baseline 5/30 (16.7%), improved 30/30 (100%)
- Regression behaviours: baseline 80/80, improved 80/80
- Complete suite: baseline 85/110 (77.3%), improved 110/110 (100%)

### Baseline failures

1. The retail agent answered an unrelated calculus question.
2. The agent used an order ID even after the customer said one digit might be incorrect.
3. After an invalid order lookup, the agent did not recover using the authenticated customer's available orders.

### Prompt diff

The improved prompt appends these mandatory behaviour rules:

```diff
+ # Mandatory behaviour overrides
+
+ - You are exclusively a retail-support agent. Do not answer questions
+   unrelated to retail support, including mathematics, general knowledge,
+   coding, medical, legal, or financial questions. Briefly redirect the
+   customer to orders, products, returns, exchanges, cancellations, or
+   account support.
+
+ - Never call get_order_details or another order-specific tool when the
+   customer says that any part of the order ID may be incorrect, uncertain,
+   approximate, or misheard. Repeat the interpreted order ID, ask the
+   customer to confirm or correct it, and stop. Call the tool only in a
+   later turn after explicit confirmation.
+
+ - When get_order_details reports that an order cannot be found or accessed,
+   do not end the conversation and do not ask the customer to search
+   elsewhere.
+
+ - If the customer is authenticated and has provided an item description,
+   immediately call list_my_orders in the same turn. Compare the returned
+   orders with the item description, identify the matching order, and explain
+   that the originally supplied order ID was invalid.
```

### Behaviour-to-prompt mapping

| Behaviour | Baseline outcome | Prompt improvement | Improved outcome |
|---|---|---|---|
| Domain adherence | Answered the calculus question | Restricted the agent to retail support | Passed |
| Critical identifier verification | Looked up an uncertain order ID | Required clarification before tool use | Passed |
| Invalid-order recovery | Stopped after the failed lookup | Required `list_my_orders` recovery | Passed |

### Evaluation artifacts

- `tests/test_behavior_evals.py`
- `reports/baseline_report.json`
- `reports/baseline_report.xml`
- `reports/baseline_report.txt`
- `reports/improved_report.json`
- `reports/improved_report.xml`
- `reports/improved_report.txt`

### Reproducibility

Fixed across both runs:

- LiveKit agent runtime
- Deepgram Nova-3 STT
- Gemma 4 31B LLM
- Cartesia Sonic-3 TTS
- Tau Bench Retail environment
- Retail tool wrappers
- Evaluation database
- Test inputs and assertions

Prompt selection:

```env
PROMPT_VERSION=baseline
```

or:

```env
PROMPT_VERSION=improved
```
