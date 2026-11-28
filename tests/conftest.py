"""Test isolation.

Disable the .env loader during tests so a developer's local .env (which may hold a real
ANTHROPIC_API_KEY or OPENAI_API_KEY) never leaks into the suite and changes behaviour.
Tests inject FakeLLM and never call the network.
"""

import os

os.environ["SCAMGUARD_DISABLE_DOTENV"] = "1"
