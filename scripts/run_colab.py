import sys
from pathlib import Path


def main() -> int:
    # Asegura que el repo (raíz) esté en sys.path
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from omr_sam2_grader.colab_entry import run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
