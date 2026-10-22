import copy
import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Settings
from .models import MODEL_ROLES, configure_offline_environment, verify_models


def pipeline_config(settings: Settings) -> dict:
    config = yaml.safe_load(Path(__file__).with_name("paddle_pipeline.yaml").read_text())
    config.update(
        batch_size=1,
        use_doc_preprocessor=True,
        use_seal_recognition=False,
        use_table_recognition=True,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=False,
        markdown_ignore_labels=[],
    )

    def model(target: dict, role: str):
        name = MODEL_ROLES[role]
        target.update(model_name=name, model_dir=str((settings.models_dir / name).resolve()))
        if "batch_size" in target:
            target["batch_size"] = 1

    layout = config["SubModules"]["LayoutDetection"]
    model(layout, "layout")
    # The vendored default is plus-L. Its per-class IDs differ from DocLayout-L.
    layout.update(threshold=settings.layout_threshold, layout_merge_bboxes_mode="union")
    pipelines = config["SubPipelines"]
    preprocessor = pipelines["DocPreprocessor"]
    preprocessor.update(batch_size=1, use_doc_orientation_classify=True, use_doc_unwarping=False)
    model(preprocessor["SubModules"]["DocOrientationClassify"], "orientation")
    general = pipelines["GeneralOCR"]
    general.update(batch_size=1, use_doc_preprocessor=False, use_textline_orientation=False)
    model(general["SubModules"]["TextDetection"], "detection")
    general["SubModules"]["TextDetection"]["limit_side_len"] = 960
    model(general["SubModules"]["TextRecognition"], "recognition")
    general["SubModules"]["TextRecognition"].update(batch_size=8, score_thresh=0.0)
    tables = pipelines["TableRecognition"]
    for name, role in {
        "TableClassification": "table_classification",
        "WiredTableStructureRecognition": "wired_structure",
        "WirelessTableStructureRecognition": "wireless_structure",
        "WiredTableCellsDetection": "wired_cells",
        "WirelessTableCellsDetection": "wireless_cells",
        "TableOrientationClassify": "orientation",
    }.items():
        model(tables["SubModules"][name], role)
    # Explicitly configure lazy backup OCR too: top-level flags do not propagate here.
    tables["SubPipelines"]["GeneralOCR"] = copy.deepcopy(general)
    return config


@dataclass
class Prediction:
    raw: dict
    image: object  # numpy BGR image; keep numpy/Paddle out of the API process
    format: str = "paddle"
    metrics: dict = field(default_factory=dict)


class PaddleEngine:
    def __init__(self, settings: Settings):
        manifest = verify_models(settings.models_dir)
        configure_offline_environment(settings.cpu_threads, settings.data_dir / "cache")
        from paddleocr import PPStructureV3

        self.pipeline = PPStructureV3(
            paddlex_config=pipeline_config(settings),
            device="cpu",
            cpu_threads=settings.cpu_threads,
            enable_mkldnn=True,
        )
        self.metadata = {
            "name": "PP-StructureV3",
            "paddleocr": importlib.metadata.version("paddleocr"),
            "paddlex": importlib.metadata.version("paddlex"),
            "paddlepaddle": importlib.metadata.version("paddlepaddle"),
            "models": MODEL_ROLES,
            "manifest": manifest,
            "cpu_threads": settings.cpu_threads,
        }

    def recognize(self, image, *, layout_threshold: float | None = None) -> Prediction:
        iterator = self.pipeline.predict_iter(
            input=image,
            layout_threshold=layout_threshold,
            use_table_orientation_classify=False,
            use_ocr_results_with_table_cells=True,
            use_e2e_wired_table_rec_model=False,
            use_e2e_wireless_table_rec_model=True,
        )
        try:
            prediction = next(iterator)
            raw = prediction.json
            processed = prediction.get("doc_preprocessor_res", {}).get("output_img", image)
            return Prediction(raw=raw, image=processed)
        finally:
            iterator.close()
