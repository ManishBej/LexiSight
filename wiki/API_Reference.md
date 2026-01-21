# API Reference

The backend exposes a REST API built with FastAPI.

## Endpoints

### 1. Generate Part 1 (Questions)

Generates clarifying questions based on the initial user query.

*   **URL**: `/generate_part1`
*   **Method**: `POST`
*   **Content-Type**: `application/json`

**Request Body**
```json
{
  "query": "My contractor delivered late material. Can I claim damages?"
}
```

**Response Body**
```json
{
  "questions": [
    "Was there a specific delivery date mentioned in the contract?",
    "Did you notify the contractor about the delay immediately?",
    "What kind of damages did you suffer due to the delay?",
    "Is there a penalty clause in your agreement?"
  ],
  "case_summary": "Extracted or generated summary of the context..."
}
```

### 2. Generate Part 2 (Analysis)

Generates the full legal analysis based on the query and answers to the questions.

*   **URL**: `/generate_part2`
*   **Method**: `POST`
*   **Content-Type**: `application/json`

**Request Body**
```json
{
  "query": "My contractor delivered late material. Can I claim damages?",
  "case_summary": "...",
  "answers": [
    "Yes, date was 1st Jan.",
    "Yes, sent an email.",
    "Project delayed by 2 weeks.",
    "Yes, 5% per week."
  ]
}
```

**Response Body**
```json
{
  "relevant_legal_sections": "Section 55 of Indian Contract Act...",
  "suggested_legal_procedures": "File a notice claiming damages...",
  "strategic_advice": "Gather evidence of the delay...",
  "estimated_outcome": "High chance of recovering damages..."
}
```

### 3. Full Analysis (Simulation)

Simulates the entire process with mock answers (useful for testing).

*   **URL**: `/full_analysis`
*   **Method**: `POST`

**Request Body**
```json
{
  "query": "..."
}
```
