
import html
import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "models/mmbert-hindi-full/best"
MAX_LENGTH = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading trained mmBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print("Using device:", device)


def prepare_input(question, context, answer):
    prefix = "Question: " + question + "\nContext: "
    answer_text = "Answer: " + answer

    answer_tokens = tokenizer(
        answer_text,
        add_special_tokens=False
    )["input_ids"]

    prefix_tokens = tokenizer(
        prefix,
        add_special_tokens=False
    )["input_ids"]

    available = MAX_LENGTH - len(answer_tokens) - 3

    if available < len(prefix_tokens):
        raise ValueError("Question or answer is too long.")

    # Same truncation strategy used during training.
    context_tokens = tokenizer(
        context,
        add_special_tokens=False
    )["input_ids"]

    context_tokens = context_tokens[
        :available - len(prefix_tokens)
    ]

    truncated_context = tokenizer.decode(
        context_tokens,
        skip_special_tokens=True
    )

    first_text = prefix + truncated_context

    encoded = tokenizer(
        first_text,
        answer_text,
        max_length=MAX_LENGTH,
        truncation="only_first",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    return encoded


def merge_spans(spans):
    if not spans:
        return []

    spans.sort()
    merged = [list(spans[0])]

    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def highlight(answer, spans):
    result = ""
    previous = 0

    for start, end in spans:
        result += html.escape(answer[previous:start])
        result += (
            '<mark style="background-color:#ffb3b3;'
            'color:#651616;padding:2px;border-radius:3px;">'
            + html.escape(answer[start:end])
            + "</mark>"
        )
        previous = end

    result += html.escape(answer[previous:])

    return (
        '<div style="font-size:20px;line-height:2;'
        'white-space:pre-wrap;">'
        + result + "</div>"
    )


def detect_hallucination(question, context, answer):
    question = question.strip()
    context = context.strip()
    answer = answer.strip()

    if not question or not context or not answer:
        return "Please fill in all three fields.", ""

    try:
        encoded = prepare_input(question, context, answer)
    except ValueError as error:
        return str(error), ""

    sequence_ids = encoded.sequence_ids(0)
    offsets = encoded["offset_mapping"][0].tolist()

    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key != "offset_mapping"
    }

    with torch.inference_mode():
        logits = model(**inputs).logits

    predictions = logits.argmax(dim=-1)[0].cpu().tolist()

    prefix_length = len("Answer: ")
    spans = []

    for label, sequence_id, (start, end) in zip(
        predictions, sequence_ids, offsets
    ):
        if sequence_id != 1:
            continue

        if end <= prefix_length:
            continue

        if label != 1:
            continue

        start = max(0, start - prefix_length)
        end = min(len(answer), end - prefix_length)

        if start < end:
            spans.append((start, end))

    spans = merge_spans(spans)

    if spans:
        status = (
            "Potential hallucination detected. "
            f"Highlighted spans: {len(spans)}"
        )
    else:
        status = (
            "No hallucinated spans detected. "
            "This does not guarantee factual correctness."
        )

    return status, highlight(answer, spans)


with gr.Blocks(title="Hindi RAG Hallucination Detector") as demo:
    gr.Markdown(
        "# Hindi RAG Hallucination Detector\n"
        "Fine-tuned mmBERT-small | Hindi answer span detection"
    )

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Hindi Question",
                lines=2
            )

            context = gr.Textbox(
                label="Retrieved Evidence / Context",
                lines=7
            )

            answer = gr.Textbox(
                label="Generated Answer",
                lines=4
            )

            detect_button = gr.Button(
                "Detect Hallucinations",
                variant="primary"
            )

        with gr.Column():
            status = gr.Textbox(
                label="Detection Result",
                interactive=False
            )

            highlighted = gr.HTML(
                label="Highlighted Answer"
            )

    detect_button.click(
        fn=detect_hallucination,
        inputs=[question, context, answer],
        outputs=[status, highlighted]
    )

demo.launch(inbrowser=True)
