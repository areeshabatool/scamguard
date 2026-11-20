"""Single gateway for every model call.

Provides retries, an on-disk response cache, and token and latency logging.
Also defines FakeLLM, the deterministic stand-in used by all tests and CI;
tests and CI never call the real API.
"""
