import os
from pathlib import Path

from .utils import ensure_dir, move_dir, make_zip, safe_remove


def organize_debug_folders(
    debug_before: str = "debug_antes",
    debug_after: str = "debug_despues",
    debug_root: str = "debug",
    keep_originals: bool = False,
) -> dict:
    """
    Reestructura:
        debug_antes/   -> debug/input/
        debug_despues/ -> debug/output/

    SIN tocar el pipeline (tu código).
    """
    debug_root = ensure_dir(debug_root)
    input_dir = Path(debug_root) / "input"
    output_dir = Path(debug_root) / "output"
    ensure_dir(input_dir)
    ensure_dir(output_dir)

    # mover
    if Path(debug_before).exists():
        move_dir(debug_before, input_dir, overwrite=True)
        # después del move_dir, el destino real queda como debug/input (no debug/input/debug_antes)
        # porque move_dir mueve la carpeta completa al destino.
        # Para que quede EXACTAMENTE debug/input/*, hacemos ajuste:
        moved_path = Path(debug_root) / "input" / Path(debug_before).name
        if moved_path.exists() and moved_path.is_dir():
            # mover contenidos un nivel arriba
            for item in moved_path.iterdir():
                item.rename(Path(debug_root) / "input" / item.name)
            safe_remove(moved_path)

    if Path(debug_after).exists():
        move_dir(debug_after, output_dir, overwrite=True)
        moved_path = Path(debug_root) / "output" / Path(debug_after).name
        if moved_path.exists() and moved_path.is_dir():
            for item in moved_path.iterdir():
                item.rename(Path(debug_root) / "output" / item.name)
            safe_remove(moved_path)

    if not keep_originals:
        safe_remove(debug_before)
        safe_remove(debug_after)

    return {
        "debug_root": str(Path(debug_root)),
        "debug_input": str(Path(debug_root) / "input"),
        "debug_output": str(Path(debug_root) / "output"),
    }


def zip_debug(debug_root: str = "debug", out_zip_base: str = "outputs/Imagenes_Diagnostico") -> str:
    """
    Comprime la carpeta debug/ en un zip:
        outputs/Imagenes_Diagnostico.zip
    """
    ensure_dir(Path(out_zip_base).parent)
    zip_path = make_zip(out_zip_base, debug_root)
    return zip_path
