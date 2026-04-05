from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from assistant_axis.internals import ActivationExtractor, ConversationEncoder, ProbingModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute role/trait vectors from generated responses.")
    parser.add_argument("--model", type=str, required=True, help="HF model id or local checkpoint path")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF model id for tokenizer (default: same as --model). Useful when --model is a local "
             "finetuned checkpoint that may lack tokenizer files.",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=["roles", "traits"],
        required=True,
        help="Whether to process data/roles/instructions or data/traits/instructions",
    )
    parser.add_argument("--questions", type=str, default="data/extraction_questions.jsonl")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-entities", type=int, default=0, help="0 means all entities")
    parser.add_argument("--max-questions", type=int, default=0, help="0 means all questions")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--include-default", action="store_true", help="Include default.json if present")
    parser.add_argument("--save-activations", action="store_true", help="Save per-sample activations too")
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=32,
        help="Number of conversations to generate and extract in one GPU batch (default: 32)",
    )
    parser.add_argument(
        "--entities-file",
        type=str,
        default=None,
        help="Optional text file with one entity name per line; only these entities are processed",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing roles/traits instructions (default: data)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help=(
            "Judge model for response quality filtering. Use an OpenAI model name "
            "(e.g. 'gpt-4o-mini') to call the OpenAI API, or a HuggingFace model id "
            "(e.g. 'meta-llama/Llama-3.1-8B-Instruct') to run locally. "
            "Set to 'none' to disable judging and include all responses (legacy). "
            "Default: gpt-4o-mini (~$1.50 for a full experiment run)."
        ),
    )
    parser.add_argument(
        "--judge-min-score",
        type=int,
        default=2,
        help="Minimum judge score (0-3) to include a response in the entity vector (default: 2).",
    )
    return parser.parse_args()


def _load_questions(path: Path, max_questions: int) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("prompt")
            if q:
                rows.append(str(q))
    if max_questions > 0:
        rows = rows[:max_questions]
    if not rows:
        raise ValueError(f"No questions found in {path}")
    return rows


def _load_instruction_file(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    instructions = obj.get("instructions")
    if instructions is None:
        instructions = obj.get("instruction", [])
    if not instructions:
        raise ValueError(f"No instructions in {path}")
    out: list[str] = []
    for item in instructions:
        if isinstance(item, dict):
            txt = item.get("pos") or item.get("neg")
            if txt:
                out.append(str(txt))
        elif isinstance(item, str):
            out.append(item)
    if not out:
        raise ValueError(f"No usable instruction strings in {path}")
    return out


def _batch_extract_vectors(
    full_conversations: list[list[dict[str, str]]],
    encoder: ConversationEncoder,
    extractor: ActivationExtractor,
    chat_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, list[int]]:
    """Batch-extract per-sample mean-pooled response vectors.

    Returns:
        tensor of shape (n_valid, n_layers, hidden_dim)
        list of original indices (into full_conversations) that produced valid vectors
    """
    acts, batch_meta = extractor.batch_conversations(full_conversations, layer=None, **chat_kwargs)
    # acts: (num_layers, batch_size, max_seq_len, hidden_size)

    sample_vectors: list[torch.Tensor] = []
    valid_indices: list[int] = []
    for i, convo in enumerate(full_conversations):
        response_indices = encoder.response_indices(convo, per_turn=False, **chat_kwargs)
        if not response_indices:
            print(f"  [warn] no response indices for batch sample {i} — skipping")
            continue

        seq_len = batch_meta["truncated_lengths"][i]
        valid_idx = [idx for idx in response_indices if idx < seq_len]
        if not valid_idx:
            print(f"  [warn] all response indices outside truncated seq (len={seq_len}) for sample {i} — skipping")
            continue

        # (num_layers, n_valid_tokens, hidden_size) → (num_layers, hidden_size)
        sample_acts = acts[:, i, valid_idx, :].mean(dim=1).cpu()
        sample_vectors.append(sample_acts)
        valid_indices.append(i)

    if not sample_vectors:
        return torch.empty(0), []
    return torch.stack(sample_vectors), valid_indices  # (n_valid, num_layers, hidden_size)


def _build_stop_ids(tokenizer: Any) -> list[int]:
    """Build the list of stop token ids for generation.

    Includes the tokenizer's eos_token_id plus any chat-end tokens
    (e.g. <|im_end|> for Qwen-family models). Passing these explicitly
    ensures generation terminates as soon as the model produces any stop
    token, without padding up to max_new_tokens.
    """
    stop: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        (stop.update(eos) if isinstance(eos, list) else stop.add(eos))
    # Explicit chat-end tokens for Qwen-family (and similar) chat models.
    for special in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(special)
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is not None and tid != unk:
            stop.add(tid)
    return sorted(stop)


def _batch_generate(
    pm: ProbingModel,
    base_conversations: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    chat_kwargs: dict[str, Any],
) -> list[str]:
    """Generate responses for a batch of conversations using left-padded batched generation."""
    prompts = [
        pm.tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        for conv in base_conversations
    ]

    orig_padding_side = pm.tokenizer.padding_side
    pm.tokenizer.padding_side = "left"
    encoded = pm.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    pm.tokenizer.padding_side = orig_padding_side

    input_ids = encoded["input_ids"].to(pm.model.device)
    attention_mask = encoded["attention_mask"].to(pm.model.device)
    n_input = input_ids.shape[1]

    stop_ids = _build_stop_ids(pm.tokenizer)
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=stop_ids,
        pad_token_id=pm.tokenizer.pad_token_id,
    )
    if do_sample:
        if temperature != 1.0:
            gen_kwargs["temperature"] = temperature
        if top_p != 1.0:
            gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = pm.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    new_token_ids = [output_ids[i][n_input:] for i in range(len(base_conversations))]
    return [pm.tokenizer.decode(tids, skip_special_tokens=True) for tids in new_token_ids]


JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation assistant. Score how well an AI response adheres to "
    "the assigned persona/role.\n"
    "Scale:\n"
    "0 = refused to engage with the role at all\n"
    "1 = deflected (said it can't be the role but offered related help)\n"
    "2 = partially plays the role (identifies as AI but adopts some role attributes)\n"
    "3 = fully plays the role as instructed\n"
    "Output ONLY a single digit 0, 1, 2, or 3. No other text."
)

JUDGE_USER_TEMPLATE = (
    "Role instruction given to the model:\n{question}\n\n"
    "Model response:\n{answer}\n\n"
    "Score:"
)

_OPENAI_MODEL_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")


def _is_openai_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _OPENAI_MODEL_PREFIXES)


def _score_responses_api(
    model_name: str,
    role_instructions: list[str],
    responses: list[str],
) -> list[int | None]:
    """Score responses via OpenAI API (e.g. gpt-4o-mini). Uses assistant_axis judge infrastructure."""
    from assistant_axis.judge import score_responses_sync

    eval_prompt_template = JUDGE_SYSTEM_PROMPT + "\n\n" + JUDGE_USER_TEMPLATE
    items = [
        {"question": instr[:300], "response": resp[:500]}
        for instr, resp in zip(role_instructions, responses)
    ]
    return score_responses_sync(
        responses=items,
        eval_prompt_template=eval_prompt_template,
        judge_model=model_name,
        max_tokens=4,
        requests_per_second=50,
        batch_size=50,
    )


def _score_responses_local(
    judge_pm: "ProbingModel",
    role_instructions: list[str],
    responses: list[str],
    chat_kwargs: dict,
    batch_size: int = 32,
) -> list[int | None]:
    """Score responses using a local HF judge model."""
    scores: list[int | None] = []
    conversations = [
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                question=instr[:300],
                answer=resp[:500],
            )},
        ]
        for instr, resp in zip(role_instructions, responses)
    ]
    for start in range(0, len(conversations), batch_size):
        batch = conversations[start : start + batch_size]
        raw_responses = _batch_generate(
            judge_pm, batch, max_new_tokens=4, temperature=0.0,
            top_p=1.0, do_sample=False, chat_kwargs=chat_kwargs,
        )
        for raw in raw_responses:
            digit = next((c for c in raw.strip() if c in "0123"), None)
            scores.append(int(digit) if digit is not None else None)
    return scores


def _score_responses(
    judge_model: str,
    role_instructions: list[str],
    responses: list[str],
    judge_pm: "ProbingModel | None" = None,
    judge_chat_kwargs: dict | None = None,
    batch_size: int = 32,
) -> list[int | None]:
    """Route to API or local judge based on model name."""
    if _is_openai_model(judge_model):
        return _score_responses_api(judge_model, role_instructions, responses)
    assert judge_pm is not None, "judge_pm required for local judge model"
    return _score_responses_local(judge_pm, role_instructions, responses, judge_chat_kwargs or {}, batch_size)


def main() -> None:
    args = parse_args()

    base_dir = Path(args.data_root) / args.kind / "instructions"
    if not base_dir.exists():
        raise ValueError(f"Instruction directory missing: {base_dir}")

    instruction_files = sorted(
        p for p in base_dir.glob("*.json")
        if not p.name.startswith("._") and not p.name.startswith(".")
    )
    if not args.include_default:
        instruction_files = [p for p in instruction_files if p.stem != "default"]
    if args.entities_file:
        requested = {
            line.strip()
            for line in Path(args.entities_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if args.include_default:
            requested.add("default")
        instruction_files = [p for p in instruction_files if p.stem in requested]
    if args.max_entities > 0:
        instruction_files = instruction_files[: args.max_entities]
    if not instruction_files:
        raise ValueError(f"No instruction files found in {base_dir}")

    questions = _load_questions(Path(args.questions), args.max_questions)

    out_dir = Path(args.output_dir)
    vectors_dir = out_dir / "vectors"
    activations_dir = out_dir / "activations"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    if args.save_activations:
        activations_dir.mkdir(parents=True, exist_ok=True)

    pm = ProbingModel(args.model, chat_model_name=args.tokenizer)

    chat_kwargs: dict[str, Any] = {}
    if pm.is_qwen:
        chat_kwargs["enable_thinking"] = False
    # Create encoder and extractor once — stateless wrappers, expensive to re-create.
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)

    # Optional judge for response quality filtering.
    judge_enabled = args.judge_model and args.judge_model.lower() != "none"
    judge_pm: ProbingModel | None = None
    judge_chat_kwargs: dict[str, Any] = {}
    if judge_enabled:
        if _is_openai_model(args.judge_model):
            print(f"Judge: OpenAI API ({args.judge_model}). Responses scoring < {args.judge_min_score} excluded.")
        else:
            print(f"Judge: local model ({args.judge_model}). Loading...")
            judge_pm = ProbingModel(args.judge_model)
            if judge_pm.is_qwen:
                judge_chat_kwargs["enable_thinking"] = False
            print(f"Judge loaded. Responses scoring < {args.judge_min_score} will be excluded from vectors.")

    summary: dict[str, Any] = {
        "model": args.model,
        "kind": args.kind,
        "n_entities": len(instruction_files),
        "n_questions": len(questions),
        "judge_model": args.judge_model,
        "judge_min_score": args.judge_min_score if args.judge_model else None,
        "entities": [],
    }

    for instruction_file in tqdm(instruction_files, desc=f"{args.kind} entities"):
        entity_name = instruction_file.stem

        # Resume: skip if vector already saved from a previous (crashed) run.
        out_path = vectors_dir / f"{entity_name}.pt"
        if out_path.exists():
            saved = torch.load(out_path, map_location="cpu", weights_only=True)
            summary["entities"].append({"entity": entity_name, "n_samples": saved["n_samples"]})
            print(f"  resume: skipping {entity_name} (vector already saved)")
            continue

        instructions = _load_instruction_file(instruction_file)

        # Build all (instruction, question) pairs upfront.
        base_convos = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
            for system_prompt in instructions
            for question in questions
        ]
        sample_meta = [
            {"instruction_idx": i, "question_idx": j}
            for i in range(len(instructions))
            for j in range(len(questions))
        ]

        sample_acts: list[torch.Tensor] = []
        kept_meta: list[dict] = []

        # Process in sub-batches to bound GPU memory usage.
        for batch_start in range(0, len(base_convos), args.gen_batch_size):
            batch_base = base_convos[batch_start : batch_start + args.gen_batch_size]
            batch_meta = list(sample_meta[batch_start : batch_start + args.gen_batch_size])

            # Pass 1: generate responses.
            responses = _batch_generate(
                pm, batch_base, args.max_new_tokens, args.temperature,
                args.top_p, args.do_sample, chat_kwargs,
            )

            # Build full conversations including the generated response.
            full_convos = [
                base + [{"role": "assistant", "content": resp}]
                for base, resp in zip(batch_base, responses)
            ]

            # Drop empty responses (model emitted nothing / immediate EOS) — no tokens
            # to pool over, so vector extraction would fail.
            non_empty = [
                (fc, m, conv[0]["content"], r)
                for fc, m, conv, r in zip(full_convos, batch_meta, batch_base, responses)
                if r.strip()
            ]
            if not non_empty:
                continue
            full_convos, batch_meta, role_instrs, responses = zip(*non_empty)
            full_convos = list(full_convos)
            batch_meta = list(batch_meta)
            role_instrs = list(role_instrs)
            responses = list(responses)

            # Judge filtering: keep only responses scoring >= judge_min_score.
            if judge_enabled:
                scores = _score_responses(
                    args.judge_model, role_instrs, responses,
                    judge_pm=judge_pm, judge_chat_kwargs=judge_chat_kwargs,
                    batch_size=args.gen_batch_size,
                )
                kept = [
                    (fc, m)
                    for fc, m, s in zip(full_convos, batch_meta, scores)
                    if s is not None and s >= args.judge_min_score
                ]
                if not kept:
                    continue
                full_convos, batch_meta = zip(*kept)
                full_convos = list(full_convos)
                batch_meta = list(batch_meta)

            # Pass 2: extract hidden-state vectors for kept samples.
            batch_vectors, valid_idx = _batch_extract_vectors(full_convos, encoder, extractor, chat_kwargs)
            if batch_vectors.numel() == 0:
                continue
            # Only keep metadata for samples that produced valid vectors.
            valid_meta = [batch_meta[j] for j in valid_idx]
            sample_acts.extend(batch_vectors.unbind(0))
            kept_meta.extend(valid_meta)

        if not sample_acts:
            continue

        stacked = torch.stack(sample_acts)  # (n_samples, n_layers, hidden_dim)
        vector = stacked.mean(dim=0)  # (n_layers, hidden_dim)

        torch.save(
            {
                "entity": entity_name,
                "kind": args.kind,
                "vector": vector,
                "n_samples": stacked.shape[0],
                "shape": list(vector.shape),
            },
            vectors_dir / f"{entity_name}.pt",
        )

        if args.save_activations:
            act_map = {
                f"i{m['instruction_idx']}_q{m['question_idx']}": a
                for m, a in zip(kept_meta, sample_acts)
            }
            torch.save(act_map, activations_dir / f"{entity_name}.pt")

        summary["entities"].append(
            {
                "entity": entity_name,
                "n_samples": int(stacked.shape[0]),
            }
        )

        # Release GPU allocator cache between entities to prevent fragmentation.
        torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()
