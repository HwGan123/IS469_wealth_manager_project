# run_benchmarks.py
import pandas as pd
import time
from graph.auditor_experiment.baseline_graph import build_baseline
from graph.auditor_experiment.baseline_graph import build_oneshot
from graph.auditor_experiment.baseline_graph import build_iterative
from agents.metrics_judge import MetricsJudge # We will define this below

# 1. Setup the Experiment
test_cases = [
    {"ticker": "NVDA", "query": "Analyze AI infrastructure spend vs revenue growth."},
    {"ticker": "TSLA", "query": "Evaluate debt-to-equity ratio impact on 2026 expansion."},
]

graphs = {
    "Exp_1_Baseline": build_baseline(),
    "Exp_2_OneShot": build_oneshot(),
    "Exp_3_Iterative": build_iterative()
}

judge = MetricsJudge()
all_results = []

# 2. Execution Loop
for name, graph in graphs.items():
    print(f"Running {name}...")
    for case in test_cases:
        start_time = time.time()
        
        # Run the Agentic Flow
        output = graph.invoke({
            "ticker": case["ticker"], 
            "query": case["query"],
            "loop_count": 0
        })
        
        latency = time.time() - start_time
        
        # 3. Metric Calculation
        # The Judge compares the final_report against the raw context
        metrics = judge.evaluate(
            report=output["final_report"], 
            context=output["context"]
        )
        
        metrics.update({
            "experiment": name,
            "ticker": case["ticker"],
            "latency": latency,
            "total_loops": output.get("loop_count", 0)
        })
        
        all_results.append(metrics)

# 4. Save and Display
df = pd.DataFrame(all_results)
# df.to_csv("experiment_results.csv", index=False)
print("\n--- Benchmark Complete ---")
print(df.groupby("experiment")[["hallucination_rate", "factual_grounding", "quality_score"]].mean())