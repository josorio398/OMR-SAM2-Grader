import os
import time

def main():
    print("--- 🎓 Calificador OMR SAM-2 ---")
    
    pdf_path = None
    
    # Bloque para Google Colab
    try:
        from google.colab import files
        print("📂 Por favor, selecciona el archivo PDF de los exámenes:")
        uploaded = files.upload()
        
        if uploaded:
            # Tomamos el primer archivo subido
            filename = list(uploaded.keys())[0]
            
            # Crear carpeta inputs si no existe
            os.makedirs("inputs", exist_ok=True)
            
            # Definir la ruta destino y moverlo
            dest_path = os.path.join("inputs", filename)
            
            # Usamos una pequeña pausa para asegurar que el sistema de archivos de Colab refresque
            with open(dest_path, "wb") as f:
                f.write(uploaded[filename])
            
            pdf_path = dest_path
            print(f"✅ Archivo '{filename}' subido y movido a /inputs")
        else:
            print("❌ No se seleccionó ningún archivo.")
            return

    except ImportError:
        # Lógica para ejecución local (PC)
        print("ℹ️ Ejecución local detectada. Buscando en carpeta 'inputs/'...")
        # Aquí puedes poner un nombre por defecto para tus pruebas locales
        pdf_path = "inputs/examen.pdf" 

    # Validación final y ejecución del procesador
    if pdf_path and os.path.exists(pdf_path):
        from src.processor import OMRProcessor
        engine = OMRProcessor()
        resultado = engine.process_pdf(pdf_path)
        print(f"✨ Proceso completado con éxito.")
        
        # Opcional: Descarga automática del resultado en Colab
        try:
            from google.colab import files
            files.download(resultado)
            print(f"📥 Descargando {resultado}...")
        except:
            pass
    else:
        print(f"❌ Error: No se pudo encontrar el archivo en {pdf_path}")

if __name__ == "__main__":
    main()
