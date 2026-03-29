import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def normalize_url_entity_id(raw_value):
    raw_value = str(raw_value).strip()
    if "/" in raw_value:
        raw_value = raw_value.split("/", 1)[0]
    if raw_value.startswith("m."):
        return "/m/" + raw_value[2:]
    return raw_value


def load_url_candidates(url_files):
    candidates = {}
    for url_file in url_files:
        df = pd.read_csv(
            url_file,
            sep="\t",
            header=None,
            names=["url", "image_ref"],
            dtype=str,
            on_bad_lines="skip",
        )
        for row in df.itertuples(index=False):
            entity_id = normalize_url_entity_id(row.image_ref)
            url = str(row.url).strip()
            if not entity_id or not url:
                continue
            candidates.setdefault(entity_id, [])
            if url not in candidates[entity_id]:
                candidates[entity_id].append(url)
    return candidates


def load_input_index(index_file):
    return pd.read_csv(
        index_file,
        sep="\t",
        header=None,
        names=["raw_name", "image_id"],
        dtype=str,
    ).dropna(subset=["raw_name", "image_id"])


def fetch_image(url, timeout):
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def build_clip_feature(
    image,
    model,
    processor,
    device,
    use_fp16,
):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if use_fp16 and device.type == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()

    with torch.no_grad():
        feature = model.get_image_features(**inputs)
        feature = torch.nn.functional.normalize(feature, dim=-1)
    return feature[0].detach().cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Build CLIP image embeddings for FB15K-style multimodal KGC."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face CLIP model name.",
    )
    parser.add_argument(
        "--input-index",
        type=str,
        default="image-graph_urls/FB15K_ImageIndex.txt",
        help="Input entity-to-image_id mapping.",
    )
    parser.add_argument(
        "--url-files",
        nargs="+",
        default=[
            "image-graph_urls/URLS_bing.txt",
            "image-graph_urls/URLS_google.txt",
            "image-graph_urls/URLS_yahoo.txt",
        ],
        help="Raw URL manifest files.",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        default="image-graph_urls/FB15K_CLIP_ImageIndex.txt",
        help="Output mapping for successfully embedded entities.",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default="image-graph_urls/FB15K_CLIP_ImageData.h5",
        help="Output H5 file for CLIP image features.",
    )
    parser.add_argument(
        "--output-meta",
        type=str,
        default="image-graph_urls/FB15K_CLIP_Metadata.json",
        help="Metadata JSON file describing the feature dump.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional entity limit for debugging. 0 means all rows.",
    )
    parser.add_argument(
        "--num-urls-per-entity",
        type=int,
        default=3,
        help="Maximum candidate URLs to try per entity.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds per image request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output H5 datasets if present.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for CLIP inference.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run CLIP image tower in fp16 on CUDA.",
    )
    args = parser.parse_args()

    input_index = load_input_index(args.input_index)
    url_candidates = load_url_candidates(args.url_files)
    if args.limit > 0:
        input_index = input_index.head(args.limit)

    output_index_path = Path(args.output_index)
    output_h5_path = Path(args.output_h5)
    output_meta_path = Path(args.output_meta)
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name)
    model.eval()
    model.to(device)
    if args.fp16 and device.type == "cuda":
        model = model.half()

    successful_rows = []
    failed_rows = []
    feature_dim = int(model.config.projection_dim)

    open_mode = "w" if args.overwrite or not output_h5_path.exists() else "a"
    with h5py.File(output_h5_path, open_mode) as h5_file:
        iterator = tqdm(
            input_index.itertuples(index=False),
            total=len(input_index),
            desc="Encoding images",
        )
        for row in iterator:
            raw_name = str(row.raw_name)
            image_id = str(row.image_id)

            if image_id in h5_file and not args.overwrite:
                successful_rows.append({"raw_name": raw_name, "image_id": image_id})
                continue

            urls = url_candidates.get(raw_name, [])[: args.num_urls_per_entity]
            if not urls:
                failed_rows.append(
                    {"raw_name": raw_name, "image_id": image_id, "reason": "no_url"}
                )
                continue

            feature = None
            last_error = "unknown"
            for url in urls:
                try:
                    image = fetch_image(url, timeout=args.request_timeout)
                    feature = build_clip_feature(
                        image=image,
                        model=model,
                        processor=processor,
                        device=device,
                        use_fp16=args.fp16,
                    )
                    break
                except (
                    requests.RequestException,
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                ) as exc:
                    last_error = type(exc).__name__

            if feature is None:
                failed_rows.append(
                    {"raw_name": raw_name, "image_id": image_id, "reason": last_error}
                )
                continue

            if image_id in h5_file:
                del h5_file[image_id]
            h5_file.create_dataset(image_id, data=feature.reshape(1, -1), dtype=np.float32)
            successful_rows.append({"raw_name": raw_name, "image_id": image_id})

    successful_df = pd.DataFrame(successful_rows, columns=["raw_name", "image_id"])
    successful_df.to_csv(
        output_index_path, sep="\t", header=False, index=False
    )

    metadata = {
        "model_name": args.model_name,
        "feature_dim": feature_dim,
        "num_successful": len(successful_rows),
        "num_failed": len(failed_rows),
        "input_index": str(Path(args.input_index)),
        "output_index": str(output_index_path),
        "output_h5": str(output_h5_path),
        "url_files": [str(Path(p)) for p in args.url_files],
    }
    output_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if failed_rows:
        failed_path = output_meta_path.with_name(output_meta_path.stem + "_failed.json")
        failed_path.write_text(json.dumps(failed_rows, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
