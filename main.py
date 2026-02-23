import os
from src.processor import OMRProcessor

def main():
    print("--- 🎓 Calificador OMR SAM-2 ---")
    
    # Intentar detectar si es Colab para pedir archivo
    try:
        from google.colab import files
        print("Sube tu archivo PDF:")
        uploaded = files.upload()
        pdf_path = list(uploaded.keys())[0]
        # Mover a la carpeta inputs
        os.rename(pdf_path, f"inputs/{pdf_path}")
        pdf_path = f"inputs/{pdf_path}"
    except:
        # Si es local, busca en inputs
        pdf_path = "inputs/examen.pdf" 

    if os.path.exists(pdf_path):
        engine = OMRProcessor()
        resultado = engine.process_pdf(pdf_path)
        print(f"✨ Proceso completado. Archivo generado: {resultado}")
    else:
        print("❌ No se encontró el archivo en la carpeta 'inputs/'.")

if __name__ == "__main__":
    main()
