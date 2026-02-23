#!/bin/bash

echo "-------------------------------------------------------"
echo "🚀 Iniciando configuración del entorno OMR SAM-2..."
echo "-------------------------------------------------------"

# 1. Crear estructura de carpetas si no existe
mkdir -p models inputs debug/antes debug/despues
echo "✅ Carpetas del proyecto creadas."

# 2. Descargar pesos del modelo SAM-2 (Versión Base Plus)
# Se descarga en la carpeta models/ para que coincida con la ruta en detector.py
if [ ! -f "models/sam2_b.pt" ]; then
    echo "⏳ Descargando pesos de SAM-2 (esto puede tardar un minuto)..."
    wget -q --show-progress -O models/sam2_b.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
    echo "✅ Modelo SAM-2 descargado exitosamente."
else
    echo "ℹ️ El modelo SAM-2 ya existe en la carpeta /models."
fi

echo "-------------------------------------------------------"
echo "✨ Configuración finalizada. ¡Listo para calificar!"
echo "-------------------------------------------------------"
