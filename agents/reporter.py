"""
Reporter Agent - Formats Final Debate Report
=============================================

"""

import os
import time
from typing import List, Optional
import google.generativeai as genai

from core.interfaces import BaseAgent, Document, DebateState


genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
_model = genai.GenerativeModel(""gemini-3.1-flash-lite")


class ReporterAgent(BaseAgent):
    """Reporter Agent formats the final debate output."""

    def __init__(self):
        super().__init__(
            name="reporter",
            model="gemini-1.5-flash",
            role="Report formatter and presenter"
        )
        print(f"✓ ReporterAgent initialized (Gemini)")

    def generate(self, query: str, context: List[Document],
                 debate_state: Optional[DebateState] = None) -> str:

        if not debate_state:
            return "[Error: No debate state to format]"

        pro_args = self._get_arguments(debate_state, 'pro')
        con_args = self._get_arguments(debate_state, 'con')
        fact_check = self._get_arguments(debate_state, 'fact_checker')
        verdict = self._get_arguments(debate_state, 'judge')

        prompt = self._build_prompt(
            query, pro_args, con_args, fact_check, verdict,
            debate_state.trust_score, debate_state.verdict
        )

        for attempt in range(3):
            try:
                response = _model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=2048,
                    )
                )
                return response.text.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    return self._fallback_report(debate_state)

    def _build_prompt(self, query, pro_args, con_args,
                      fact_check, verdict, trust_score, verdict_type) -> str:
        return f"""You are a professional report writer. Create a concise executive summary.

QUERY: {query}

PRO ARGUMENTS (summary):
{pro_args[:500]}

CON ARGUMENTS (summary):
{con_args[:500]}

FACT-CHECK:
{fact_check[:300] if fact_check else 'Not available'}

VERDICT: {verdict_type}
TRUST SCORE: {trust_score:.1f}%

Write an executive summary with these sections:
1. VERDICT SUMMARY (2-3 sentences)
2. KEY FINDINGS (3-4 bullet points)
3. RECOMMENDATION (specific action)
4. RISK FACTORS (2-3 risks)
5. CONFIDENCE ASSESSMENT

RESPONSE:"""

    def _get_arguments(self, debate_state: DebateState, agent_name: str) -> str:
        all_args = []
        for round_data in debate_state.rounds:
            if agent_name in round_data:
                content = round_data[agent_name]['content']
                if not content.startswith("[Error:"):
                    all_args.append(content)
        return "\n\n".join(all_args) if all_args else ""

    def _fallback_report(self, debate_state: DebateState) -> str:
        return (
            f"{'='*70}\nEXECUTIVE SUMMARY\n{'='*70}\n"
            f"Query: {debate_state.query}\n"
            f"Verdict: {debate_state.verdict}\n"
            f"Trust Score: {debate_state.trust_score:.1f}%\n"
            f"{'='*70}"
        )
