"""
Con Agent - Skeptical Risk Analyst
===================================
Identifies risks and bearish factors with evidence.

"""

import os
import time
from typing import List, Optional
import google.generativeai as genai

from core.interfaces import BaseAgent, Document, DebateState


genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
_model = genai.GenerativeModel("gemini-2.0-flash")


class ConAgent(BaseAgent):
    """Con Agent argues AGAINST the investment/decision."""

    def __init__(self):
        super().__init__(
            name="con",
            model="gemini-1.5-flash",
            role="Skeptical risk analyst"
        )
        print(f"✓ ConAgent initialized (Gemini)")

    def generate(self, query: str, context: List[Document],
                 debate_state: Optional[DebateState] = None) -> str:

        context_text = self.format_context(context)
        is_rebuttal = debate_state and debate_state.current_round > 0

        if is_rebuttal:
            pro_args = self._get_pro_arguments(debate_state)
            prompt = self._build_rebuttal_prompt(query, context_text, pro_args)
        else:
            prompt = self._build_initial_prompt(query, context_text)

        for attempt in range(3):
            try:
                response = _model.generate_content(
                    self.system_prompt + "\n\n" + prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.4,
                        max_output_tokens=500,
                    )
                )
                return response.text.strip()
            except Exception as e:
                if attempt < 2:
                    print(f"⚠ Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
                else:
                    return f"[Error: Con Agent failed: {e}]"

    def _build_initial_prompt(self, query: str, context: str) -> str:
        return f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

YOUR TASK: Present 3-5 BEARISH arguments highlighting risks and concerns.
Cite sources like [Source: filename] for every claim.

RESPONSE:"""

    def _build_rebuttal_prompt(self, query: str, context: str, pro_args: str) -> str:
        return f"""CONTEXT DOCUMENTS:
{context}

USER QUESTION: {query}

PRO AGENT'S ARGUMENTS:
{pro_args}

YOUR TASK: Counter the Pro Agent's optimism with evidence-based concerns.
Identify overlooked risks and challenge overly bullish assumptions.

RESPONSE:"""

    def _get_pro_arguments(self, debate_state: DebateState) -> str:
        if debate_state.rounds:
            last_round = debate_state.rounds[-1]
            return last_round.get('pro', {}).get('content', 'No arguments yet')
        return 'No arguments yet'
