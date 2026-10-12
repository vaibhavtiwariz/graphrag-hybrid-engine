from __future__ import annotations

import argparse
import json

import httpx

DEFAULT_QUESTION = (
    "Which supplier's machining defect caused the AtlasArm actuator overheating incident, "
    "and which batch was affected?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the running GraphRAG API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    health = httpx.get(f"{args.base_url}/health", timeout=30).json()
    print("health:", json.dumps(health))

    response = httpx.post(f"{args.base_url}/query", json={"question": args.question}, timeout=600)
    response.raise_for_status()
    payload = response.json()

    print("\nanswer:", payload["answer"])
    print("\ncitations:")
    for citation in payload.get("citations", []):
        print(f"  - [{citation['source']}] {citation['title']} (score={citation['score']})")
    print(
        f"\niterations={payload['iterations']} used_web={payload['used_web']} "
        f"latency_ms={payload['latency_ms']}"
    )


if __name__ == "__main__":
    main()
