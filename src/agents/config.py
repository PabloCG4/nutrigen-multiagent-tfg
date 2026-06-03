# --- LLM CONFIGURATION ---
NUTRITIONIST_LLM_MODEL: str = "gpt-4o-mini"       # Phase 1 — allergen/ingredient assessment
NUTRITIONIST_CRITIQUE_LLM_MODEL: str = "gpt-4o-mini"   # Phase 2/3 — faster default (override if needed)
CHEF_LLM_MODEL: str = "gpt-4o-mini"
MEDIATOR_LLM_MODEL: str = "gpt-4o-mini"                # Integrates + rewrites 3 recipes (fast default)
