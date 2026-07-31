## Overview

Built a LiveKit-based retail voice agent using the Tau Bench Retail environment.

Voice pipeline:

- Deepgram Nova-3 for STT
- Gemma 4 31B for reasoning and tool use
- Cartesia Sonic-3 for TTS
- LiveKit turn detection and audio enhancement

## Implementation approach

- Wrapped the Tau Bench Retail tools in a modular `RetailToolset`.
- Added email and name-plus-ZIP authentication.
- Supported order lookup, discovery, cancellation, modification, returns, exchanges, product lookup, address updates, calculations, and human transfer.
- Created a separate reproducible evaluation database with customers having multiple orders.
- Added deterministic tool-call assertions and LLM-judged response assertions using the LiveKit test framework.
- Stored baseline and improved results as JSON, XML, and text reports.

## Three observed baseline failures

1. The retail agent answered unrelated questions such as a calculus problem.
2. It used an order ID even when the customer said a digit might be incorrect.
3. It failed to recover from an invalid order ID despite having the item description and access to the customer's orders.

## Prompt improvements

- Restricted the agent to retail-support requests.
- Required clarification before using uncertain order IDs.
- Required recovery through `list_my_orders` after an invalid lookup when identifying details are available.

## Results

| Version | Passed | Failed |
|---|---:|---:|
| Baseline | 8 | 3 |
| Improved prompt | 11 | 0 |

The LLM, STT, TTS, tools, database, and tests remained fixed. Only the prompt changed.

## Trade-offs

- Used a cascaded STT–LLM–TTS pipeline instead of speech-to-speech for better observability.
- Used a small targeted evaluation suite rather than running the entire benchmark due to the two-day timeline.
- Used a custom evaluation database without modifying the original Tau Bench database.
- Combined deterministic assertions with LLM judging only where semantic flexibility was necessary.

## Future improvements

- Run each test multiple times to measure variance.
- Add audio-level tests for noisy speech, accents, interruptions, and alphanumeric transcription.
- Measure latency across STT, LLM, tool use, and TTS.
- Add more multi-intent tasks and write-action confirmation scenarios.
- Run a broader subset of official Tau Bench Retail tasks.

See `CHANGELOG.md` for the exact prompt diff and `reports/` for evaluation artifacts.
