import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiments.architecture_workflow.workflow_experiment import (
    RunResult, NodeResult, print_node_sequence_table, print_comparison_table
)

data = json.loads(Path("experiments/architecture_workflow/results.json").read_text())

results = []
for entry in data:
    r = RunResult(workflow=entry["workflow"], query=entry["query"])
    r.total_latency_ms        = entry["total_latency_ms"]
    r.total_input_tokens      = entry["total_input_tokens"]
    r.total_output_tokens     = entry["total_output_tokens"]
    r.judge_scores            = entry["judge_scores"]
    r.success                 = entry["success"]
    r.error                   = entry.get("error")
    r.market_context_skipped  = entry.get("market_context_skipped", False)
    r.orchestrator_iterations = entry.get("orchestrator_iterations", 0)
    r.audit_retries           = entry.get("audit_retries", 0)
    r.nodes = [
        NodeResult(
            name          = n["name"],
            latency_ms    = n["latency_ms"],
            input_tokens  = n["input_tokens"],
            output_tokens = n["output_tokens"],
            tool_calls    = n["tool_calls"],
            tool_errors   = n["tool_errors"],
            success       = n["success"],
            error         = n.get("error"),
        )
        for n in entry.get("nodes", [])
    ]
    results.append(r)

print_node_sequence_table(results)
print_comparison_table(results)
