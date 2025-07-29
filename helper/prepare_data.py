#!/usr/bin/env python3
"""
PDF page ➜ PNG ➜ docTR OCR ➜ <stem>_words.txt + <stem>_rows.txt
"""
import openai
api_key = ""

import base64
import json
import os
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from doctr.models import ocr_predictor          # HF-hosted weights
from surya.table_rec import TableRecPredictor
import colorsys
import random

def _generate_palette(unique_labels: list[int]) -> dict[int, tuple[float, float, float]]:
    """Return {label: (r,g,b)} with each channel in 0-1."""
    n = len(unique_labels) or 1
    return {
        lbl: colorsys.hsv_to_rgb(random.random(), 0.65, 0.92)      # already 0-1 floats
        for idx, lbl in enumerate(sorted(unique_labels))
    }


def _load_boxes(txt: Path):
    lines = [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
    n = int(lines[0])
    coords = list(map(float, "".join(lines[1:]).split(",")))
    boxes  = [coords[i:i+4] for i in range(0, len(coords), 4)]
    if len(boxes) != n:
        print(f"⚠️  {txt.name}: header says {n} boxes, decoded {len(boxes)}")
    return boxes

def overlay_boxes(img, img_path : Path, words, rows, save_overlay: bool = True):
    w, h   = img.size
    # print(f"📷 {img.name} ({w}x{h})")
    # words  = _load_boxes(words)
    # rows   = _load_boxes(rows)

    # auto-scale if boxes look normalised (all ≤ 1.2)
    # def rescale(bxs):
    #     if all(v <= 1.2 for v in sum(bxs, [])):
    #         return [[int(x1*w), int(y1*h), int(x2*w), int(y2*h)] for x1,y1,x2,y2 in bxs]
    #     return [[int(x1), int(y1), int(x2), int(y2)] for x1,y1,x2,y2 in bxs]

    # words, rows = rescale(words), rescale(rows)

    if not words and not rows:
        print("❌ no boxes to draw – check your _words.txt / _rows.txt files")
        return

    plt.figure(figsize=(10, h / w * 10))
    plt.imshow(img)
    ax = plt.gca()

    for x1, y1, x2, y2 in words:
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor='red', lw=1))
    for x1, y1, x2, y2 in rows:
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor='blue', lw=2))

    ax.axis('off')
    if save_overlay:
        out_png = img_path.with_name(img_path.stem + "_overlay.png")
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        print("✅ overlay saved →", out_png)
    # plt.show()

def overlay_words(img, img_path: Path, words, save_overlay: bool = True):
    w, h   = img.size
    # print(f"📷 {img_path.name} ({w}x{h})")
    # words  = _load_boxes(words)

    # auto-scale if boxes look normalised (all ≤ 1.2)
    # def rescale(bxs):
    #     if all(v <= 1.2 for v in sum(bxs, [])):
    #         return [[int(x1*w), int(y1*h), int(x2*w), int(y2*h)] for x1,y1,x2,y2 in bxs]
    #     return [[int(x1), int(y1), int(x2), int(y2)] for x1,y1,x2,y2 in bxs]

    # words, rows = rescale(words), rescale(rows)

    if not words:
        print("❌ no boxes to draw – check your _words.txt files")
        return

    plt.figure(figsize=(10, h / w * 10))
    plt.imshow(img)
    ax = plt.gca()

    for x1, y1, x2, y2 in words:
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor='red', lw=1))
    ax.axis('off')
    if save_overlay:
        out_png = img_path.with_name(img_path.stem + "_overlay_words.png")
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        print("✅ overlay saved →", out_png)
    # plt.show()

def overlay_words_clustered(img, img_path: Path, words, clusters, save_overlay: bool = True):
    w, h   = img.size
    # print(f"📷 {img_path.name} ({w}x{h})")
    # words  = _load_boxes(words)

    # auto-scale if boxes look normalised (all ≤ 1.2)
    # def rescale(bxs):
    #     if all(v <= 1.2 for v in sum(bxs, [])):
    #         return [[int(x1*w), int(y1*h), int(x2*w), int(y2*h)] for x1,y1,x2,y2 in bxs]
    #     return [[int(x1), int(y1), int(x2), int(y2)] for x1,y1,x2,y2 in bxs]

    # words, rows = rescale(words), rescale(rows)

    if not words:
        print("❌ no boxes to draw – check your _words.txt files")
        return
    
    unique_clusters = set(clusters)
    palette = _generate_palette(sorted(unique_clusters))

    plt.figure(figsize=(10, h / w * 10))
    plt.imshow(img)
    ax = plt.gca()

    for (x1, y1, x2, y2), cluster_idx in zip(words, clusters):
        color = palette.get(cluster_idx, (1, 0, 0))
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor=color, lw=1))
    ax.axis('off')
    if save_overlay:
        out_png = img_path.with_name(img_path.stem + "_overlay_words_clusters.png")
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
        print("✅ overlay saved →", out_png)
    # plt.show()

def gptv_clustering(words, img_path, eps, model = "gpt-4o-mini-instruct-vision", temperature = 0.0):
    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    # normalise: ensure (x1,y1,x2,y2,label)
    prepared = []
    for i, wb in enumerate(words):
        if len(wb) == 5 and isinstance(wb[4], str):
            prepared.append((*wb[:4], wb[4]))
        else:
            prepared.append((*wb[:4], f"id_{i}"))

    # encode PNG as data URL
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(str(img_path))[1][1:] or "png"
    data_url = f"data:image/{ext};base64,{b64}"

    # shared JSON schema
    schema = {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"word": {"type": "string"}, "cluster": {"type": "integer"}},
                    "required": ["word", "cluster"],
                },
            }
        },
        "required": ["clusters"],
    }

    # routing flags
    needs_rsp = model.endswith("-deep-research") or model.startswith("o1")
    temp_ok = model not in {"o4-mini", "o4-mini-2025-04-16"}
    func = "cluster_rows"

    # ------------------------------------------------------------------
    # 1) RESPONSES ENDPOINT (deep‑research & o1)
    # ------------------------------------------------------------------
    if needs_rsp:
        tools = [{"type": "function", "name": func, "description": "Cluster word boxes by row", "parameters": schema}]
        prompt_text = (
            "You are a meticulous document‑layout reasoning assistant. "
            "Think step‑by‑step about vertical alignment, row structure, and textual context. "
            "Group boxes whose centre‑y positions lie on the same logical text row. "
            "Double‑check your reasoning before responding. "
            "Do not leave any word boxes unclustered."
            "Return *only* a call to `cluster_rows`.\n\n"
            f"Boxes JSON: {json.dumps(prepared)}"
        )
        payload = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": data_url},
            ],
        }]
        kw = dict(model=model, input=payload, tools=tools, tool_choice={"name": func})
        if temp_ok:
            kw["temperature"] = temperature
        rsp = client.responses.create(**kw)
        clusters_json = json.loads(rsp.tools[0].arguments)["clusters"]

    # ------------------------------------------------------------------
    # 2) CHAT COMPLETIONS ENDPOINT
    # ------------------------------------------------------------------
    else:
        tools = [{"type": "function", "function": {"name": func, "description": "Cluster word boxes by row", "parameters": schema}}]
        sys_prompt = (
            "You are a meticulous table‑layout reasoning assistant.\n"
            "Think step‑by‑step about row grouping, and text semantics and return cluster indices corresponding to word boxes such that word boxes belonging to same cluster belong to the same row.\n"
            "Do not leave any word boxes unclustered. the number of cluster indices must match the number of word boxes.\n"
            "When confident, call the function `cluster_rows` with JSON containing a `clusters` array.\n"
            "Do not output anything else.\n"
            "IMPORTANT : Disregard row lines in the image and semantically predict wether rows need to be merged like the Unit and Type in the topmost row need to be in same row because they are representing the same row (header of the table).Use this line of reasoning to cluster the word boxes.\n"
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "Here are the boxes to cluster:\n```json\n" + json.dumps(prepared) + "\n```"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]},
        ]
        kw = dict(model=model, messages=messages, tools=tools, tool_choice={"type": "function", "function": {"name": func}})
        if temp_ok:
            kw["temperature"] = temperature
        
        res = []
        i = 0
        while len(res) != len(words):
            chat = client.chat.completions.create(**kw)
            clusters_json = json.loads(chat.choices[0].message.tool_calls[0].function.arguments)["clusters"]
            res = [c["cluster"] for c in clusters_json]
            i += 1
            print(f"🔄 Iteration {i}: found {len(res)} clusters")
    
    return res

def gptz_clustering(words,
                    img_path: str,
                    model: str = "o4-mini-instruct-vision",
                    temperature: float = 0.0,
                    max_tries: int = 7) -> list[int]:
    """
    Returns a list `cluster_id[i]` for every `words[i]`.
    Any GPT failure after `max_tries` raises RuntimeError.
    """
    # ---------- prep -------------------------------------------------
    client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    boxes = [( *wb[:4],  wb[4] if len(wb) == 5 and isinstance(wb[4], str) else f"id_{i}")
             for i, wb in enumerate(words)]

    # image → data-URL (once!)
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(img_path)[1][1:] or "png"
    data_url = f"data:image/{ext};base64,{b64}"

    schema = {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word":    {"type": "string"},
                        "cluster": {"type": "integer"}
                    },
                    "required": ["word", "cluster"]
                }
            }
        },
        "required": ["clusters"]
    }
    tool = {"type": "function",
            "function": {"name": "cluster_rows",
                         "description": "Cluster word boxes by rows",
                         "parameters": schema}}

    # ---------- build multimodal chat prompt ------------------------
    sys_prompt = (
        "You cluster word-boxes into table rows. "
        "Return cluster indices so each word has exactly one cluster.\n"
        "Think step-by-step about row grouping, and text semantics. Total Lease, Amendment, Charge should be grouped with Lease Id Lease From Lease To.\n"
        "Respond ONLY via a tool call."
    )
    user_content = [
        {"type": "text",
         "text": "Boxes JSON:\n```json\n" + json.dumps(boxes, ensure_ascii=False) + "\n```"},
        {"type": "image_url", "image_url": {"url": data_url}}
    ]
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_content}]

    # ---------- retry loop ------------------------------------------
    kwargs = dict(model=model,
                  messages=messages,
                  tools=[tool],
                  tool_choice={"type": "function", "function": {"name": "cluster_rows"}})
    if model != "o4-mini-instruct-vision":
        kwargs["temperature"] = temperature

    for attempt in range(1, max_tries + 1):
        chat = client.chat.completions.create(**kwargs)
        clusters_raw = json.loads(
            chat.choices[0].message.tool_calls[0].function.arguments
        )["clusters"]

        clusters = [c["cluster"] for c in clusters_raw]
        if len(clusters) == len(words):
            return clusters

        print(f"🔄 retry {attempt}: model returned {len(clusters)} clusters")

    raise RuntimeError("Model failed to return the correct number of clusters.")


def gpt_clustering(
    words,                 # list of (x0, y0, x1, y1, text?)
    img_path,
    model="o3",            # fast vision models; "o3-pro" ⇒ Responses API
    temperature=0.0,
    max_tries=7,
    api_key=None,
):
    client   = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
    endpoint = "responses" if model.startswith("o3-pro") else "chat"   # :contentReference[oaicite:4]{index=4}

    # ─── prepare image & boxes ────────────────────────────────────────────
    boxes = [(*wb[:4], wb[4] if len(wb) >= 5 and isinstance(wb[4], str) else f"id_{i}")
             for i, wb in enumerate(words)]
    with open(img_path, "rb") as f:
        ext = os.path.splitext(img_path)[1][1:] or "png"
        data_url = f"data:image/{ext};base64,{base64.b64encode(f.read()).decode()}"
    boxes_json = json.dumps(boxes, ensure_ascii=False)

    # ─── build tool schema (chat ≠ responses) ────────────────────────────
    if endpoint == "chat":
        tool_spec = {
            "type": "function",
            "function": {
                "name": "cluster_rows",
                "description": "Cluster word boxes by rows",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "word":    {"type": "string"},
                                    "cluster": {"type": "integer"},
                                },
                                "required": ["word", "cluster"],
                            },
                        }
                    },
                    "required": ["clusters"],
                },
            },
        }
        tool_choice = {"type": "function", "function": {"name": "cluster_rows"}}
    else:  # Responses – flattened form   :contentReference[oaicite:5]{index=5}
        tool_spec = {
            "type": "function",
            "name": "cluster_rows",
            "description": "Cluster word boxes by rows",
            "parameters": {
                "type": "object",
                "properties": {
                    "clusters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word":    {"type": "string"},
                                "cluster": {"type": "integer"},
                            },
                            "required": ["word", "cluster"],
                        },
                    }
                },
                "required": ["clusters"],
            },
        }
        tool_choice = {"type": "function", "name": "cluster_rows"}

    sys_prompt = (
        "You cluster word‑boxes into table rows. Return a cluster index for each "
        "word. Think about horizontal alignment and semantics. Respond ONLY with "
        "a tool call."
    )

    # ─── request bodies ──────────────────────────────────────────────────
    if endpoint == "chat":
        req = dict(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"Boxes JSON:\n```json\n{boxes_json}\n```"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            tools=[tool_spec],
            tool_choice=tool_choice,
            temperature=temperature,
        )
        call = client.chat.completions.create
    else:  # Responses API
        req = dict(
            model=model,
            instructions=sys_prompt,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text",
                     "text": f"Boxes JSON:\n```json\n{boxes_json}\n```"},
                    {"type": "input_image",
                     "image_url": data_url},       # string, not object  :contentReference[oaicite:6]{index=6}
                ],
            }],
            tools=[tool_spec],
            tool_choice=tool_choice,
            temperature=temperature,
        )
        call = client.responses.create

    # ─── retry loop ───────────────────────────────────────────────────────
    for attempt in range(1, max_tries + 1):
        rsp = call(**req)
        tool_calls = (rsp.choices[0].message.tool_calls
                      if endpoint == "chat"
                      else rsp.output[0].tool_calls)
        clusters_raw = json.loads(tool_calls[0].function.arguments)["clusters"]
        clusters = [c["cluster"] for c in clusters_raw]
        if len(clusters) == len(words):
            return clusters
        print(f"🔄 retry {attempt}: got {len(clusters)} clusters, expected {len(words)}")

    raise RuntimeError("Model failed to return the correct number of clusters.")


# ---------- CLI ----------
cli = argparse.ArgumentParser()
cli.add_argument("pdf"); cli.add_argument("page", type=int)       # 1-based
cli.add_argument("--dpi", type=int, default=600)
cli.add_argument("--out", type=Path, default=Path("."))
cli.add_argument("--gpu", action="store_true", help="run on CUDA")
args = cli.parse_args()

pdf  = Path(args.pdf).expanduser().resolve()
out  = args.out.expanduser(); out.mkdir(parents=True, exist_ok=True)
stem = f"{pdf.stem}_p{args.page}"

# ---------- 1 ▸ rasterise ----------
png = out / f"{stem}.png"
if not png.exists():
    pil = convert_from_path(pdf, dpi=args.dpi,
                            first_page=args.page, last_page=args.page,
                            fmt="png")[0]
    pil.save(png)
else:
    pil = Image.open(png)

# ---------- 2 ▸ docTR predictor ----------
predictor = ocr_predictor(                         # builds detector+recogniser
    det_arch="db_resnet50",                       # HF ckpt: smartmind/db_resnet50 :contentReference[oaicite:3]{index=3}
    reco_arch="crnn_vgg16_bn",                    # HF ckpt: crnn_vgg16_bn :contentReference[oaicite:4]{index=4}
    pretrained=True,
    assume_straight_pages=True,
    export_as_straight_boxes=True,
)
if args.gpu:
    predictor = predictor.to("cuda")              # move AFTER creation :contentReference[oaicite:5]{index=5}

doc = predictor([np.asarray(pil)])                # returns a Document object
page = doc.export()["pages"][0]                   # JSON-like dict

# collect word boxes
word_boxes = [
    [int(w["geometry"][0][0] * pil.width),
     int(w["geometry"][0][1] * pil.height),
     int(w["geometry"][1][0] * pil.width),
     int(w["geometry"][1][1] * pil.height)]
    for blk in page["blocks"]                    # words are nested: blocks ▸ lines ▸ words :contentReference[oaicite:6]{index=6}
    for line in blk["lines"]
    for w in line["words"]
]

word_boxes_with_text = []
for blk in page["blocks"]:
    for line in blk["lines"]:
        for w in line["words"]:
            x0 = int(w["geometry"][0][0] * pil.width)
            y0 = int(w["geometry"][0][1] * pil.height)
            x1 = int(w["geometry"][1][0] * pil.width)
            y1 = int(w["geometry"][1][1] * pil.height)
            text = w["value"]
            word_boxes_with_text.append((x0, y0, x1, y1, text))

# ---------- 3 ▸ Surya table rows ----------
rows = TableRecPredictor(device="cuda" if args.gpu else "cpu")([pil])[0].rows
row_boxes = [c.bbox for c in rows]

# ------------ 3.1 ▸ omit boxes outside the table ----------
min_x = np.inf
min_y = np.inf
max_x = -1
max_y = -1

for row in row_boxes:
    min_x = min(min_x, row[0])
    min_y = min(min_y, row[1])
    max_x = max(max_x, row[2])
    max_y = max(max_y, row[3])

# filter word boxes that are outside the table area
word_boxes = [
    box for box in word_boxes
    if min_x <= box[0] <= max_x and min_y <= box[1] <= max_y and
       min_x <= box[2] <= max_x and min_y <= box[3] <= max_y
]

word_boxes_with_text = [
    (x0, y0, x1, y1, text) for (x0, y0, x1, y1, text) in word_boxes_with_text
    if min_x <= x0 <= max_x and min_y <= y0 <= max_y and
       min_x <= x1 <= max_x and min_y <= y1 <= max_y
]

# ---------- 4 ▸ write files ----------
def dump(tag, boxes):
    flat = ", ".join(map(str, sum(boxes, [])))
    Path(out / f"{stem}_{tag}.txt").write_text(f"{len(boxes)}\n{flat}")

def dump_clusters(clusters):
    flat = ", ".join(map(str, clusters))
    Path(out / f"{stem}_clusters.txt").write_text(f"{len(clusters)}\n{flat}")


dump_words = [[i[1], i[3]] for i in word_boxes]
dump_rows = [[i[1], i[3]] for i in row_boxes]

dump("words", dump_words)
dump("rows",  dump_rows)

overlay_boxes(pil, png, word_boxes, row_boxes)
overlay_words(pil, png, word_boxes)

clusters = gpt_clustering(word_boxes_with_text, png, model="o3", temperature=1, api_key=api_key)

for i in range(4, 8):
    clusters[i] = clusters[8]

print(len(clusters), "clusters found", "for", len(word_boxes), "words")
# print(clusters)
dump_clusters(clusters)
overlay_words_clustered(pil, png, word_boxes, clusters)



print(f"✅  PNG + txt files saved in {out}")
print(f"   Words: {len(word_boxes)}  |  Rows: {len(row_boxes)}")
