from __future__ import annotations

from typing import Optional, Tuple

from src.agents.chef import ChefAgent
from src.agents.mediator import MediatorAgent
from src.agents.nutritionist import NutritionistAgent

_nutritionist: Optional[NutritionistAgent] = None
_chef: Optional[ChefAgent] = None
_mediator: Optional[MediatorAgent] = None


def get_nutritionist_agent() -> NutritionistAgent:
    global _nutritionist
    if _nutritionist is None:
        _nutritionist = NutritionistAgent()
    return _nutritionist


def get_chef_agent() -> ChefAgent:
    global _chef
    if _chef is None:
        _chef = ChefAgent()
    return _chef


def get_mediator_agent() -> MediatorAgent:
    global _mediator
    if _mediator is None:
        _mediator = MediatorAgent()
    return _mediator


def get_agents() -> Tuple[NutritionistAgent, ChefAgent, MediatorAgent]:
    """
    Centralized agent singletons used by both the LangGraph pipeline and backend endpoints.
    """
    return get_nutritionist_agent(), get_chef_agent(), get_mediator_agent()

