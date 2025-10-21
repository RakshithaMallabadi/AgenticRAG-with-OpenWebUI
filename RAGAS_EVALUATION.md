# RAGAs Evaluation Framework Documentation

## Overview

This document describes the **RAGAs (Retrieval-Augmented Generation Assessment)** integration in the OpenWebUI-RAG project. RAGAs is a framework for evaluating RAG systems using multiple metrics to assess answer quality, context relevance, and faithfulness.

## What is RAGAs?

RAGAs provides a set of metrics to evaluate RAG systems:

1. **Faithfulness** (0-1): Measures if the generated answer is grounded in the given context
   - Score of 1.0 means the answer is fully supported by the context
   - Lower scores indicate hallucination or unsupported claims

2. **Answer Relevancy** (0-1): Measures if the answer actually addresses the question
   - Score of 1.0 means the answer directly answers the question
   - Lower scores indicate off-topic or incomplete answers

3. **Context Precision** (0-1): Measures if the retrieved contexts are relevant
   - Score of 1.0 means all retrieved chunks are relevant
   - Lower scores indicate irrelevant or noisy retrievals

4. **Context Recall** (0-1): Measures if all relevant information was retrieved
   - **Requires ground truth reference answer**
   - Score of 1.0 means all necessary context was retrieved
   - Lower scores indicate missing information

## Architecture

### Components

```
┌──────────────────────────────────────────────────────────────┐
│                     RAGAs Evaluation                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐ │
│  │ LlamaIndex  │────▶ │ RAGAs        │────▶ │ Metrics    │ │
│  │ LLM Wrapper │      │ Evaluators   │      │ Dashboard  │ │
│  └─────────────┘      └──────────────┘      └────────────┘ │
│         │                    │                      │        │
│         │                    │                      │        │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐ │
│  │ Embeddings  │      │ Faithfulness │      │ JSON       │ │
│  │ Wrapper     │      │ Relevancy    │      │ Response   │ │
│  └─────────────┘      │ Precision    │      └────────────┘ │
│                       │ Recall       │                      │
│                       └──────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

### Configuration

RAGAs is initialized in `app.py` with LlamaIndex wrappers:

```python
# Wrap LlamaIndex LLM and embeddings for RAGAs
ragas_llm = LlamaIndexLLMWrapper(llm)
ragas_embeddings = LlamaIndexEmbeddingsWrapper(embed_model)
```

This allows RAGAs to use the same LLM (Ollama) and embedding model (sentence-transformers/all-MiniLM-L6-v2) that the RAG system uses.

## API Endpoints

### 1. RAGAs Status

**GET** `/ragas/status`

Returns the current status of RAGAs evaluation framework.

**Example:**
```bash
curl http://localhost:4000/ragas/status | jq
```

**Response:**
```json
{
  "enabled": true,
  "metrics_available": [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall"
  ],
  "metrics_description": {
    "faithfulness": "Measures if answer is grounded in given context (0-1)",
    "answer_relevancy": "Measures if answer addresses the question (0-1)",
    "context_precision": "Measures if retrieved contexts are relevant (0-1)",
    "context_recall": "Measures if all relevant info is retrieved (0-1, needs ground_truth)"
  },
  "llm_configured": true,
  "embeddings_configured": true,
  "status": "operational"
}
```

### 2. Single Query Evaluation

**POST** `/evaluate`

Evaluates a single RAG query end-to-end.

**Parameters:**
- `question` (required): The question to evaluate
- `ground_truth` (optional): Reference answer for comparison
- `top_k` (optional, default=5): Number of contexts to retrieve

**Example:**
```bash
curl -X POST "http://localhost:4000/evaluate?question=What%20is%20the%20document%20about?&top_k=5" \
  -H "Content-Type: application/json" | jq
```

**With Ground Truth:**
```bash
curl -X POST "http://localhost:4000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who issued the Abu Dhabi Procurement Standards?",
    "ground_truth": "The document was issued by the Department of Government Support - Abu Dhabi",
    "top_k": 5
  }' | jq
```

**Response Structure:**
```json
{
  "question": "Who issued the document?",
  "answer": "The document was issued by...",
  "contexts": [
    "Context 1 text...",
    "Context 2 text..."
  ],
  "ground_truth": "Reference answer",
  "num_contexts": 5,
  "metrics": {
    "faithfulness": 0.95,
    "answer_relevancy": 0.88,
    "context_precision": 0.92,
    "context_recall": 0.85
  },
  "sources": [
    {
      "index": 1,
      "source": "document.pdf",
      "score": 0.87,
      "text_preview": "First 200 chars..."
    }
  ],
  "evaluation_summary": {
    "total_metrics": 4,
    "average_score": 0.90,
    "has_errors": false
  }
}
```

### 3. Batch Evaluation

**POST** `/evaluate/batch`

Evaluates multiple queries in batch for systematic testing.

**Request Body:**
```json
{
  "questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ],
  "ground_truths": [
    "Reference answer 1",
    "Reference answer 2",
    "Reference answer 3"
  ],
  "contexts": null,
  "top_k": 5
}
```

**Example:**
```bash
curl -X POST "http://localhost:4000/evaluate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "What is the main topic?",
      "Who are the key stakeholders?"
    ],
    "ground_truths": [
      "The main topic is procurement standards",
      "Key stakeholders include government departments"
    ],
    "top_k": 5
  }' | jq
```

**Response Structure:**
```json
{
  "total_questions": 3,
  "successful_evaluations": 3,
  "failed_evaluations": 0,
  "results": [
    {
      "question": "Question 1?",
      "answer": "Generated answer...",
      "contexts": ["Context 1", "Context 2"],
      "ground_truth": "Reference answer 1",
      "num_contexts": 5,
      "metrics": {
        "faithfulness": 0.95,
        "answer_relevancy": 0.88,
        "context_precision": 0.92,
        "context_recall": 0.85
      }
    },
    ...
  ],
  "average_metrics": {
    "avg_faithfulness": 0.93,
    "avg_answer_relevancy": 0.86,
    "avg_context_precision": 0.90,
    "avg_context_recall": 0.84
  },
  "evaluation_success": true
}
```

## Use Cases

### 1. Development Testing

Test your RAG system during development:

```bash
# Quick test with a known question
curl -X POST "http://localhost:4000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main requirements?",
    "ground_truth": "The main requirements are A, B, and C",
    "top_k": 3
  }' | jq '.metrics'
```

### 2. Benchmarking

Create a test suite and evaluate in batch:

```bash
# Save test suite to file
cat > test_suite.json << EOF
{
  "questions": [
    "Question 1 about the document?",
    "Question 2 about specific details?",
    "Question 3 about requirements?"
  ],
  "ground_truths": [
    "Expected answer 1",
    "Expected answer 2",
    "Expected answer 3"
  ],
  "top_k": 5
}
EOF

# Run batch evaluation
curl -X POST "http://localhost:4000/evaluate/batch" \
  -H "Content-Type: application/json" \
  -d @test_suite.json > evaluation_results.json

# View average metrics
jq '.average_metrics' evaluation_results.json
```

### 3. Continuous Monitoring

Monitor RAG quality over time:

```bash
# Periodic evaluation script
while true; do
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  curl -X POST "http://localhost:4000/evaluate/batch" \
    -H "Content-Type: application/json" \
    -d @test_suite.json > "eval_${TIMESTAMP}.json"
  
  # Extract average scores
  jq '.average_metrics' "eval_${TIMESTAMP}.json"
  
  sleep 3600  # Run every hour
done
```

### 4. A/B Testing

Compare different RAG configurations:

```bash
# Test with different top_k values
for k in 3 5 7 10; do
  echo "Testing with top_k=$k"
  curl -X POST "http://localhost:4000/evaluate?question=Test&top_k=$k" | \
    jq ".metrics"
done
```

## Interpreting Metrics

### Faithfulness (0-1)

- **> 0.9**: Excellent - answer is well-grounded
- **0.7-0.9**: Good - mostly supported by context
- **0.5-0.7**: Fair - some unsupported claims
- **< 0.5**: Poor - significant hallucination

**How to improve:**
- Increase context window
- Use stricter prompts
- Add citation requirements

### Answer Relevancy (0-1)

- **> 0.9**: Excellent - directly answers question
- **0.7-0.9**: Good - mostly on-topic
- **0.5-0.7**: Fair - partially relevant
- **< 0.5**: Poor - off-topic

**How to improve:**
- Better retrieval (higher top_k)
- Re-ranking enabled
- Clearer system prompts

### Context Precision (0-1)

- **> 0.9**: Excellent - all contexts relevant
- **0.7-0.9**: Good - mostly relevant
- **0.5-0.7**: Fair - some noise
- **< 0.5**: Poor - too much irrelevant data

**How to improve:**
- Enable re-ranking
- Adjust similarity threshold
- Better chunking strategy

### Context Recall (0-1)

**Note:** Requires ground truth reference answer

- **> 0.9**: Excellent - all info retrieved
- **0.7-0.9**: Good - most info present
- **0.5-0.7**: Fair - missing some details
- **< 0.5**: Poor - significant gaps

**How to improve:**
- Increase top_k for retrieval
- Better embeddings
- Improve chunking overlap

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: RAG Quality Check

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start Services
        run: docker compose up -d
      
      - name: Wait for Services
        run: sleep 120
      
      - name: Run RAGAs Evaluation
        run: |
          curl -X POST "http://localhost:4000/evaluate/batch" \
            -H "Content-Type: application/json" \
            -d @test_suite.json > results.json
      
      - name: Check Quality Thresholds
        run: |
          AVG=$(jq '.average_metrics.avg_faithfulness' results.json)
          if (( $(echo "$AVG < 0.7" | bc -l) )); then
            echo "Quality check failed: faithfulness=$AVG"
            exit 1
          fi
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: results.json
```

## Best Practices

### 1. Create a Test Suite

Maintain a curated set of questions and reference answers:

```json
{
  "test_cases": [
    {
      "id": "TC001",
      "question": "What is X?",
      "ground_truth": "X is defined as...",
      "expected_min_faithfulness": 0.8
    },
    {
      "id": "TC002",
      "question": "How does Y work?",
      "ground_truth": "Y works by...",
      "expected_min_faithfulness": 0.85
    }
  ]
}
```

### 2. Track Metrics Over Time

Log evaluation results to detect degradation:

```bash
# Log to CSV
echo "timestamp,faithfulness,relevancy,precision,recall" > metrics.csv

curl -X POST "http://localhost:4000/evaluate?question=Test" | \
  jq -r '[now, .metrics.faithfulness, .metrics.answer_relevancy, 
          .metrics.context_precision, .metrics.context_recall] | @csv' >> metrics.csv
```

### 3. Set Quality Gates

Define minimum acceptable thresholds:

```python
MIN_FAITHFULNESS = 0.75
MIN_RELEVANCY = 0.70
MIN_PRECISION = 0.65

def check_quality(metrics):
    return (
        metrics['faithfulness'] >= MIN_FAITHFULNESS and
        metrics['answer_relevancy'] >= MIN_RELEVANCY and
        metrics['context_precision'] >= MIN_PRECISION
    )
```

### 4. Use Ground Truth Wisely

- Start with a small set of high-quality reference answers
- Expand gradually as you identify edge cases
- Review and update ground truths regularly

## Troubleshooting

### RAGAs Not Available

```bash
# Check status
curl http://localhost:4000/ragas/status

# If disabled, rebuild with RAGAs
docker compose build fastapi
docker compose up -d
```

### Low Scores

1. **Low Faithfulness**: 
   - Check if context contains answer
   - Review LLM prompt engineering
   - Increase context provided

2. **Low Relevancy**:
   - Review retrieval effectiveness
   - Check if question is ambiguous
   - Enable re-ranking

3. **Low Precision**:
   - Enable Cohere re-ranking
   - Adjust similarity cutoff
   - Improve document chunking

4. **Low Recall**:
   - Increase `top_k` value
   - Review chunking strategy
   - Check embedding quality

### Evaluation Timeout

If evaluation takes too long:

```bash
# Reduce top_k
curl -X POST "http://localhost:4000/evaluate?question=Test&top_k=3"

# Or increase timeout in docker-compose.yml
environment:
  - OLLAMA_REQUEST_TIMEOUT=300
```

## Advanced Usage

### Custom Metrics

RAGAs supports custom metrics. To add your own:

```python
from ragas.metrics import Metric

class CustomMetric(Metric):
    def calculate(self, row):
        # Your custom logic
        return score
```

### Integration with Phoenix

RAGAs evaluations are automatically traced in Phoenix:

```bash
# View in Phoenix UI
open http://localhost:6006

# Filter for evaluation traces
# They appear with span names: "ragas.evaluate"
```

### Programmatic Access

Use the Python API directly:

```python
import httpx

def evaluate_query(question: str, ground_truth: str = None):
    response = httpx.post(
        "http://localhost:4000/evaluate",
        params={"question": question, "ground_truth": ground_truth}
    )
    return response.json()

# Example
result = evaluate_query(
    "What is the main topic?",
    "The main topic is procurement standards"
)
print(f"Faithfulness: {result['metrics']['faithfulness']:.2f}")
```

## References

- RAGAs Documentation: https://docs.ragas.io/
- RAGAs GitHub: https://github.com/explodinggradients/ragas
- LlamaIndex Integration: https://docs.llamaindex.ai/en/stable/examples/evaluation/ragas.html
- Paper: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (https://arxiv.org/abs/2309.15217)

## Related Documentation

- [Main README](README.md) - Project overview
- [Phoenix Observability](PHOENIX_OBSERVABILITY.md) - Tracing and monitoring
- [CrewAI Implementation](CREWAI_IMPLEMENTATION.md) - Multi-agent system
- [OpenWebUI Integration](OPENWEBUI_CREWAI_INTEGRATION.md) - Chat interface

## Summary

RAGAs provides a comprehensive evaluation framework for assessing RAG system quality. By measuring faithfulness, relevancy, precision, and recall, you can:

- **Monitor** quality over time
- **Benchmark** different configurations
- **Identify** weak points in retrieval or generation
- **Improve** system performance systematically

Use the `/evaluate` endpoint for quick tests and `/evaluate/batch` for comprehensive quality assessments.

