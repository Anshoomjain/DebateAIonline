"""
Pro Agent - Optimistic Financial Analyst
=========================================
Finds and presents bullish arguments with evidence.
Now powered by Google Gemini API (free tier).
"""

import os
import time
from typing import List, Optional
import google.generativeai as genai

from core.interfaces import BaseAgent, Document, DebateState


# Initialize Gemini client once at module level
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
_model = genai.GenerativeModel(""gemini-3.1-flash-lite")


class ProAgent(BaseAgent):
    """Pro Agent argues FOR the investment/decision."""

    def __init__(self):
        super().__init__(
            name="pro",
            model="gemini-1.5-flash",
            role="Optimistic financial analyst"
        )
        print(f"✓ ProAgent initialized (Gemini)")

    def generate(self, query: str, context: List[Document],
                 debate_state: Optional[DebateState] = None) -> str:

        context_text = self.format_context(context)
        is_rebuttal = debate_state and debate_state.current_round > 0

        if is_rebuttal:
            con_args = self._get_con_arguments(debate_state)
            prompt = self._build_rebuttal_prompt(query, context_text, con_args)
        else:
            prompt = self._build_initial_prompt(query, context_text)

        for attempt in range(3):
            try:
                response = _model.generate_content(
                    self.system_prompt + "\n\n" + prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=1500,
                    )
                )
                return response.text.strip()
            except Exception as e:
                if attempt < 2:
                    print(f"⚠ Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
                else:
                    return f"[Error: Pro Agent failed: {e}]"

    def _build_initial_prompt(self, query: str, context: str) -> str:
        return f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

YOUR TASK: Present 3-5 BULLISH arguments supporting this investment/decision.
Cite sources like [Source: filename] for every claim.

RESPONSE:"""

    def _build_rebuttal_prompt(self, query: str, context: str, con_args: str) -> str:
        return f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

CON AGENT'S ARGUMENTS:
{con_args}

YOUR TASK: Respond to the Con Agent's concerns while reinforcing your bullish case.
Address their specific points with evidence from the context.

RESPONSE:"""

    def _get_con_arguments(self, debate_state: DebateState) -> str:
        if debate_state.rounds:
            last_round = debate_state.rounds[-1]
            return last_round.get('con', {}).get('content', 'No arguments yet')
        return 'No arguments yet'
