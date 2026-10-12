from __future__ import annotations

import asyncio
from typing import Any

from app.config import get_settings
from app.core.llm import llm_endpoint


class OllamaDeepEvalLLM:
    """DeepEval-compatible LLM wrapper around any OpenAI-compatible endpoint."""

    def __init__(self) -> None:
        try:
            from deepeval.models import DeepEvalBaseLLM
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install -e '.[eval]'") from exc

        from openai import OpenAI

        base_url, api_key = llm_endpoint()
        model_name = get_settings().llm_model
        client_factory = OpenAI

        class _Wrapper(DeepEvalBaseLLM):
            def __init__(self) -> None:
                self._client = client_factory(
                    base_url=base_url, api_key=api_key or "none", timeout=180
                )
                self._model = model_name

            def load_model(self):
                return self._client

            def generate(self, prompt: str) -> str:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content or ""

            async def a_generate(self, prompt: str) -> str:
                return await asyncio.to_thread(self.generate, prompt)

            def get_model_name(self) -> str:
                return model_name

        self._wrapper = _Wrapper()

    def measure_correctness(self, cases: list[dict], rows: list[dict]) -> dict[str, Any]:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        metric = GEval(
            name="Correctness",
            criteria=(
                "Determine whether the actual output is factually correct compared to the "
                "expected output and grounded in the retrieval context."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            model=self._wrapper,
            threshold=0.5,
        )

        scores = []
        for row in rows:
            test_case = LLMTestCase(
                input=row["user_input"],
                actual_output=row["response"],
                expected_output=row["reference"],
                retrieval_context=row["retrieved_contexts"][:4],
            )
            metric.measure(test_case)
            scores.append(metric.score or 0.0)

        return {"correctness_avg": round(sum(scores) / max(len(scores), 1), 4), "n": len(scores)}


def run_deepeval(cases: list[dict], rows: list[dict]) -> dict[str, Any]:
    return OllamaDeepEvalLLM().measure_correctness(cases, rows)
