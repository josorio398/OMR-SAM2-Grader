import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("🔎 Smoke test: imports básicos...")

    try:
        import torch  # noqa: F401
        print("✅ torch OK")
    except Exception as e:
        print(f"❌ torch FAIL: {e}")
        return 1

    try:
        import fitz  # PyMuPDF # noqa: F401
        print("✅ pymupdf (fitz) OK")
    except Exception as e:
        print(f"❌ pymupdf FAIL: {e}")
        return 1

    try:
        import cv2  # noqa: F401
        print("✅ opencv-python OK")
    except Exception as e:
        print(f"❌ opencv FAIL: {e}")
        return 1

    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # noqa: F401
        print("✅ transformers OK")
    except Exception as e:
        print(f"❌ transformers FAIL: {e}")
        return 1

    try:
        from ultralytics import SAM  # noqa: F401
        print("✅ ultralytics SAM OK")
    except Exception as e:
        print(f"❌ ultralytics FAIL: {e}")
        return 1

    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        print("✅ rapidocr-onnxruntime OK")
    except Exception as e:
        print(f"❌ rapidocr FAIL: {e}")
        return 1

    print("\n✅ Smoke test completado: entorno listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
