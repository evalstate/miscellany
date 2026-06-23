export const storyGepaTraceData = {
  "run": "20260623-074130-optimize",
  "candidates": [
    {
      "id": "candidate-0001",
      "iteration": 1,
      "score": 0.2888,
      "promptWords": 3,
      "storyWords": 1122
    },
    {
      "id": "candidate-0002",
      "iteration": 2,
      "score": 0.2888,
      "promptWords": 3,
      "storyWords": 1100
    },
    {
      "id": "candidate-0003",
      "iteration": 3,
      "score": 0.94,
      "promptWords": 16,
      "storyWords": 104
    },
    {
      "id": "candidate-0004",
      "iteration": 4,
      "score": 0.94,
      "promptWords": 16,
      "storyWords": 107
    },
    {
      "id": "candidate-0005",
      "iteration": 5,
      "score": 0.4388,
      "promptWords": 3,
      "storyWords": 1196
    },
    {
      "id": "candidate-0006",
      "iteration": 6,
      "score": 0.94,
      "promptWords": 16,
      "storyWords": 99
    },
    {
      "id": "candidate-0007",
      "iteration": 7,
      "score": 0.94,
      "promptWords": 16,
      "storyWords": 109
    },
    {
      "id": "candidate-0008",
      "iteration": 8,
      "score": 0.94,
      "promptWords": 16,
      "storyWords": 107
    },
    {
      "id": "candidate-0009",
      "iteration": 9,
      "score": 0.9512,
      "promptWords": 13,
      "storyWords": 98
    },
    {
      "id": "candidate-0010",
      "iteration": 10,
      "score": 0.9512,
      "promptWords": 13,
      "storyWords": 110
    },
    {
      "id": "candidate-0011",
      "iteration": 11,
      "score": 0.9512,
      "promptWords": 13,
      "storyWords": 87
    },
    {
      "id": "candidate-0012",
      "iteration": 12,
      "score": 0.9737,
      "promptWords": 7,
      "storyWords": 97
    },
    {
      "id": "candidate-0013",
      "iteration": 13,
      "score": 0.9737,
      "promptWords": 7,
      "storyWords": 115
    }
  ],
  "steps": [
    {
      "id": "call-0001",
      "label": "Reflection 1",
      "candidateId": "candidate-0002",
      "iteration": 2,
      "currentPrompt": "Write a story.",
      "proposedPrompt": "Write a short children's story (under 120 words) that includes Maya, a lighthouse, and a lamp.",
      "score": 0.2888,
      "scores": {
        "requiredItems": 0.0,
        "storyLength": 0.0,
        "promptLength": 0.9625
      },
      "promptWords": 3,
      "storyWords": 1100,
      "asi": [
        "Missing required: Maya, lamp, lighthouse",
        "Story too long: 1100 words",
        "Keep prompt short"
      ]
    },
    {
      "id": "call-0002",
      "label": "Reflection 2",
      "candidateId": "candidate-0005",
      "iteration": 5,
      "currentPrompt": "Write a story.",
      "proposedPrompt": "Write a short children's story under 120 words that includes Maya, a lighthouse, and a lamp.",
      "score": 0.4388,
      "scores": {
        "requiredItems": 0.33333333333333337,
        "storyLength": 0.0,
        "promptLength": 0.9625
      },
      "promptWords": 3,
      "storyWords": 1196,
      "asi": [
        "Missing required: Maya, lamp",
        "Story too long: 1196 words",
        "Keep prompt short"
      ]
    },
    {
      "id": "call-0003",
      "label": "Reflection 3",
      "candidateId": "candidate-0008",
      "iteration": 8,
      "currentPrompt": "Write a short children's story under 120 words that includes Maya, a lighthouse, and a lamp.",
      "proposedPrompt": "Write a children's story ≤120 words with Maya, a lighthouse, and a lamp.",
      "score": 0.94,
      "scores": {
        "requiredItems": 1.0,
        "storyLength": 1.0,
        "promptLength": 0.8
      },
      "promptWords": 16,
      "storyWords": 107,
      "asi": [
        "All rubric items pass",
        "Story length passes",
        "Compress the prompt"
      ]
    },
    {
      "id": "call-0004",
      "label": "Reflection 4",
      "candidateId": "candidate-0011",
      "iteration": 11,
      "currentPrompt": "Write a children's story ≤120 words with Maya, a lighthouse, and a lamp.",
      "proposedPrompt": "Write ≤120-word story with Maya, lighthouse, lamp.",
      "score": 0.9512,
      "scores": {
        "requiredItems": 1.0,
        "storyLength": 1.0,
        "promptLength": 0.8375
      },
      "promptWords": 13,
      "storyWords": 87,
      "asi": [
        "All rubric items pass",
        "Story length passes",
        "Compress again"
      ]
    }
  ],
  "best": {
    "prompt": "Write ≤120-word story with Maya, lighthouse, lamp.",
    "score": 0.9737
  }
} as const
