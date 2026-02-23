import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Ruta del archivo PDF")
    args = parser.parse_args()

    pdf_path = args.file

    if os.path.exists(pdf_path):
        from src.processor import OMRProcessor
        procesador = OMRProcessor()
        procesador.process_pdf(pdf_path)
    else:
        print(f"\n❌ Error: No se encontró el archivo {pdf_path}")

if __name__ == "__main__":
    main()
