import argparse
import logging
import os
import sys
from pathlib import Path

from .config import Settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(prog="ocr-service")
    commands = parser.add_subparsers(dest="command", required=True)
    api = commands.add_parser("api", help="Run the HTTP API and browser client")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    commands.add_parser("worker", help="Run the single persistent CPU OCR worker")
    prepare = commands.add_parser("prepare-models", help="Prepare or verify local models")
    prepare.add_argument(
        "--download",
        action="store_true",
        default=os.environ.get("OCR_ALLOW_MODEL_DOWNLOAD", "false").lower() == "true",
        help="Explicitly allow model downloads",
    )
    prepare.add_argument("--import-from", type=Path, help="Copy a prepared offline model directory")
    commands.add_parser("check-models", help="Check all model files and SHA256 hashes offline")
    args = parser.parse_args()
    settings = Settings()
    if args.command == "api":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)
    elif args.command == "worker":
        from .worker import run_worker

        run_worker(settings)
    else:
        from .models import ModelsUnavailable, copy_local_models, prepare_models, verify_models

        try:
            for backend in settings.backends:
                if backend == "paddle":
                    if args.command == "check-models":
                        manifest = verify_models(settings.models_dir)
                    elif args.import_from:
                        copy_local_models(args.import_from, settings.models_dir)
                        manifest = verify_models(settings.models_dir)
                    else:
                        manifest = prepare_models(settings.models_dir, allow_download=args.download)
                elif backend in {"falcon", "nemotron"}:
                    from .hf_models import prepare_hf, verify_hf

                    manifest = (
                        verify_hf(backend, settings)
                        if args.command == "check-models"
                        else prepare_hf(
                            backend, settings, download=args.download, source=args.import_from
                        )
                    )
                elif backend == "kdl":
                    from .kdl_models import prepare_kdl, verify_kdl

                    manifest = (
                        verify_kdl(settings)
                        if args.command == "check-models"
                        else prepare_kdl(settings, download=args.download, source=args.import_from)
                    )
                else:
                    from .vlm_models import prepare_vlm, verify_vlm

                    manifest = (
                        verify_vlm(backend, settings)
                        if args.command == "check-models"
                        else prepare_vlm(
                            backend, settings, download=args.download, source=args.import_from
                        )
                    )
                logging.info(
                    "Verified %s local models: %s", backend, manifest.get("revision", "Paddle")
                )
        except ModelsUnavailable as error:
            logging.error("%s", error)
            sys.exit(1)
        logging.info("Verified models in %s", settings.models_dir)


if __name__ == "__main__":
    main()
