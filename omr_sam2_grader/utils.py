import os
import shutil
from pathlib import Path


def ensure_dir(path: str | Path) -> str:
    """Crea carpeta si no existe y retorna ruta string."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def safe_remove(path: str | Path) -> None:
    """Elimina archivo/carpeta si existe."""
    p = Path(path)
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except Exception:
            pass


def move_dir(src: str | Path, dst: str | Path, overwrite: bool = True) -> None:
    """Mueve carpeta src → dst. Si overwrite=True, borra dst antes."""
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.exists():
        return
    if overwrite and dst_p.exists():
        shutil.rmtree(dst_p, ignore_errors=True)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_p), str(dst_p))


def make_zip(zip_base_name: str, folder_to_zip: str | Path) -> str:
    """
    Crea zip a partir de una carpeta.
    zip_base_name: sin extensión, ej: "outputs/Imagenes_Diagnostico"
    Retorna: ruta del zip generado.
    """
    folder_to_zip = str(folder_to_zip)
    zip_path = shutil.make_archive(zip_base_name, "zip", root_dir=folder_to_zip)
    return zip_path


def in_colab() -> bool:
    """Detecta si está corriendo en Google Colab."""
    return "COLAB_GPU" in os.environ or "google.colab" in str(os.environ)
