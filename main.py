import os
import argparse
from src.processor import OMRProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Ruta del archivo PDF")
    args = parser.parse_args()

    print("--- 🎓 Calificador OMR SAM-2 ---")
    
    pdf_path = args.file

    # Si no se pasa argumento, busca el archivo por defecto en inputs/
    if not pdf_path:
        pdf_path = "inputs/examen.pdf"

    if os.path.exists(pdf_path):
        engine = OMRProcessor()
        resultado = engine.process_pdf(pdf_path)
        print(f"✨ Proceso completado con éxito.")
    else:
        print(f"❌ Error: No se encontró el archivo en {pdf_path}")

if __name__ == "__main__":
    main()
