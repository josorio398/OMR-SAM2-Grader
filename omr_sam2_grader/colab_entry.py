from pathlib import Path

from .utils import ensure_dir
from .pipeline_exact import run_pipeline_exact
from .postprocess import organize_debug_folders, zip_debug


def run():
    """
    Punto de entrada pensado para Colab.
    - Prepara carpetas
    - Ejecuta pipeline exacto
    - Reorganiza debug a debug/input y debug/output
    - Crea outputs/ con copias limpias del excel/zip y descarga
    """
    # carpetas base
    ensure_dir("outputs")
    ensure_dir("debug/input")
    ensure_dir("debug/output")

    # 1) Ejecuta tu pipeline (descarga Excel + ZIP como en tu código)
    result = run_pipeline_exact()

    # 2) Reorganiza carpetas debug_antes/debug_despues → debug/input/debug/output
    organize_debug_folders(
        debug_before="debug_antes",
        debug_after="debug_despues",
        debug_root="debug",
        keep_originals=False,
    )

    # 3) Genera ZIP “limpio” de debug/ dentro de outputs/
    debug_zip = zip_debug(debug_root="debug", out_zip_base="outputs/Imagenes_Diagnostico")

    # 4) Mueve/copias outputs “limpios”
    excel_src = Path(result["excel"])
    zip_src = Path(result["zip"])

    excel_dst = Path("outputs") / excel_src.name
    zip_dst = Path("outputs") / zip_src.name

    # Copias si existen
    try:
        if excel_src.exists():
            excel_dst.write_bytes(excel_src.read_bytes())
        if zip_src.exists():
            zip_dst.write_bytes(zip_src.read_bytes())
    except Exception:
        pass

    # 5) Descargas extra (además de las que ya hace tu pipeline)
    try:
        from google.colab import files
        if excel_dst.exists():
            files.download(str(excel_dst))
        if Path(debug_zip).exists():
            files.download(str(debug_zip))
    except Exception:
        pass

    print("\n✅ Listo:")
    print(f" - Excel: {excel_dst if excel_dst.exists() else excel_src}")
    print(f" - Debug ZIP: {debug_zip}")
    print(" - Debug folders: debug/input y debug/output")

    return {
        "excel": str(excel_dst if excel_dst.exists() else excel_src),
        "debug_zip": str(debug_zip),
        "df": result.get("df"),
        "debug_dir": "debug",
    }
