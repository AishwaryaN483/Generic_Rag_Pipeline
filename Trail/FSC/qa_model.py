from transformers import pipeline
import torch

_QA_CACHE = {}

def _get_device():
    return 0 if torch.cuda.is_available() else -1


def get_best_answer_extractive(question, contexts, model_type):
    if "extractive" not in _QA_CACHE:
        _QA_CACHE["extractive"] = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=_get_device()
        )

    qa = _QA_CACHE["extractive"]

    best = {"score": -1}
    for c in contexts:
        out = qa(question=question, context=c["chunk"])
        if out["score"] > best["score"]:
            best = out

    return {
        "answer": best["answer"],
        "score": best["score"]
    }


def get_best_answer_generative(question, contexts, model_type):
    model_map = {
        "gpt2": "distilgpt2",
        "smollm": "HuggingFaceTB/SmolLM2-135M-Instruct"
    }

    if model_type not in _QA_CACHE:
        _QA_CACHE[model_type] = pipeline(
            "text-generation",
            model=model_map[model_type],
            device=_get_device()
        )

    generator = _QA_CACHE[model_type]

    MAX_CONTEXT_CHARS = 1200

    context_text = " ".join(c["chunk"] for c in contexts)
    context_text = context_text.replace("\n", " ").strip()
    context_text = context_text[:MAX_CONTEXT_CHARS]

    prompt = (
        f"Context: {context_text}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    out = generator(
        prompt,
        truncation=True,
        max_new_tokens=120,
        return_full_text=True
    )[0]["generated_text"]

    answer = out.split("Answer:")[-1].strip()

    return {
        "answer": answer,
        "score": None
    }
