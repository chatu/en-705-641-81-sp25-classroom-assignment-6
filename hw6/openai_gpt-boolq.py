"""Few-shot prompting of an OpenAI completion model on BoolQ (assignment section 3.3.4).

Builds one in-context prompt from 8 BoolQ demonstrations, balanced 4 yes / 4 no and
interleaved so neither the label distribution nor the final demonstration's label biases
the prediction, then scores a held-out subset of the validation split.

    export OPENAI_API_KEY=<your key>
    python openai_gpt-boolq.py --model gpt-3.5-turbo-instruct --num-eval 30

The model is a flag because the model the handout names, gpt-3.5-turbo-instruct, now
returns a deprecation error from the Completions endpoint, as does davinci. Pass whichever
model the instructor confirms; the prompt format and scoring do not change with it.
"""

import argparse
import os
import random

from datasets import load_dataset
from openai import OpenAI

# the handout specifies 8 demonstrations, half of each label
NUM_DEMOS = 8
LABEL_WORD = {True: "yes", False: "no"}


def build_prompt(demos, question, passage):
    """Render the few-shot prompt. Demonstrations and the query share one format so the
    model only has to continue the pattern."""
    blocks = []
    for demo in demos:
        blocks.append(
            f"Passage: {demo['passage']}\n"
            f"Question: {demo['question']}?\n"
            f"Answer: {LABEL_WORD[bool(demo['answer'])]}"
        )
    blocks.append(f"Passage: {passage}\nQuestion: {question}?\nAnswer:")
    return "\n\n".join(blocks)


def pick_demonstrations(train_split, rng):
    """Sample NUM_DEMOS/2 of each label, then interleave them yes, no, yes, no, ...

    Sampling each label separately is what guarantees the 50/50 split the handout asks for;
    drawing 8 rows at random would inherit BoolQ's 62% yes skew. Interleaving puts neither
    label last, which is what the handout means by the recency bias.
    """
    per_label = NUM_DEMOS // 2
    yes_pool = [r for r in train_split if bool(r["answer"])]
    no_pool = [r for r in train_split if not bool(r["answer"])]
    yes = rng.sample(yes_pool, per_label)
    no = rng.sample(no_pool, per_label)

    interleaved = []
    for y, n in zip(yes, no):
        interleaved.append(y)
        interleaved.append(n)
    return interleaved


def parse_answer(text):
    """Map a completion onto a boolean, or None when the model did not answer the question.

    Returning None rather than guessing keeps an unparseable generation from being scored as
    a coin flip, which would make the accuracy depend on the label balance of the eval subset.
    """
    first = text.strip().lower().split()
    if not first:
        return None
    token = first[0].strip(".,:;!\"'")
    if token in ("yes", "true"):
        return True
    if token in ("no", "false"):
        return False
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("BOOLQ_MODEL", "gpt-3.5-turbo-instruct"))
    parser.add_argument("--num-eval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-passage-chars", type=int, default=1200,
                        help="truncate demonstration passages so 8 of them fit the context")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    client = OpenAI()

    print(f"Loading BoolQ ...")
    boolq = load_dataset("google/boolq")

    # keep the demonstrations short enough that eight of them plus the query fit comfortably
    train_rows = [
        {"question": r["question"], "answer": r["answer"],
         "passage": str(r["passage"])[: args.max_passage_chars]}
        for r in boolq["train"].select(range(2000))
    ]
    demos = pick_demonstrations(train_rows, rng)
    print(f"Demonstration labels in order: {[LABEL_WORD[bool(d['answer'])] for d in demos]}")

    eval_rows = boolq["validation"].select(range(args.num_eval))

    correct = 0
    unparsed = 0
    for i, row in enumerate(eval_rows):
        prompt = build_prompt(demos, row["question"], str(row["passage"]))
        response = client.completions.create(
            model=args.model,
            prompt=prompt,
            # greedy: this is a classification task, so sampling only adds variance
            temperature=0,
            max_tokens=4,
            stop=["\n"],
        )
        raw = response.choices[0].text
        predicted = parse_answer(raw)
        gold = bool(row["answer"])
        if predicted is None:
            unparsed += 1
        elif predicted == gold:
            correct += 1
        print(f"[{i + 1:>3}/{args.num_eval}] predicted={str(predicted):>5} gold={str(gold):>5} "
              f"raw={raw.strip()!r}")

    accuracy = correct / len(eval_rows)
    majority = sum(1 for r in eval_rows if bool(r["answer"])) / len(eval_rows)
    print("-" * 60)
    print(f"model              : {args.model}")
    print(f"demonstrations     : {NUM_DEMOS} ({NUM_DEMOS // 2} yes, {NUM_DEMOS // 2} no, interleaved)")
    print(f"evaluated on       : {len(eval_rows)} BoolQ validation instances")
    print(f"unparseable        : {unparsed}")
    print(f"accuracy           : {accuracy:.4f} ({correct}/{len(eval_rows)})")
    print(f"majority-class rate: {majority:.4f}  <- the number accuracy has to beat to mean anything")


if __name__ == "__main__":
    main()
