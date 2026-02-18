from __future__ import annotations

import argparse

from stemmacodicum.infrastructure.parsers.docling_adapter import DoclingRuntimeOptions


def add_docling_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--docling-auto-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-tune docling resource settings from detected hardware (default: enabled).",
    )
    parser.add_argument(
        "--docling-use-threaded-pipeline",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force threaded or standard PDF pipeline mode. Omit to let auto-tune decide.",
    )
    parser.add_argument(
        "--docling-device",
        default=None,
        help="Inference device override: auto, cpu, mps, xpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--docling-threads",
        type=int,
        default=None,
        help="CPU thread count override for docling accelerator options.",
    )
    parser.add_argument(
        "--docling-layout-batch-size",
        type=int,
        default=None,
        help="Layout stage batch size override.",
    )
    parser.add_argument(
        "--docling-ocr-batch-size",
        type=int,
        default=None,
        help="OCR stage batch size override.",
    )
    parser.add_argument(
        "--docling-table-batch-size",
        type=int,
        default=None,
        help="Table stage batch size override.",
    )
    parser.add_argument(
        "--docling-queue-max-size",
        type=int,
        default=None,
        help="Inter-stage queue limit override for threaded pipeline.",
    )
    parser.add_argument(
        "--docling-log-settings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log resolved docling runtime settings for each PDF extraction (default: enabled).",
    )


def get_docling_runtime_options(args: argparse.Namespace) -> DoclingRuntimeOptions:
    return DoclingRuntimeOptions(
        auto_tune=args.docling_auto_tune,
        use_threaded_pipeline=args.docling_use_threaded_pipeline,
        device=args.docling_device,
        num_threads=args.docling_threads,
        layout_batch_size=args.docling_layout_batch_size,
        ocr_batch_size=args.docling_ocr_batch_size,
        table_batch_size=args.docling_table_batch_size,
        queue_max_size=args.docling_queue_max_size,
        log_settings=args.docling_log_settings,
    )
