# ============================================================================
# DOCKERFILE OPTIMIZADO PARA RADIOLOGY AI BACKEND
# Puerto 8002 para evitar conflictos
# ============================================================================

# Usar imagen base Python 3.11 slim (más ligera)
FROM python:3.11-slim

# Información del mantenedor
LABEL maintainer="Radiology AI Team"
LABEL description="Backend de análisis radiológico con IA - TorchXRayVision"
LABEL version="2.0.0"

# ============================================================================
# VARIABLES DE ENTORNO GLOBALES
# ============================================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8002
ENV HOST=0.0.0.0

# Variables específicas para modelos de IA
ENV MODEL_CACHE_DIR=/app/models
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface

# Variables críticas para TorchXRayVision
ENV TORCHXRAYVISION_CACHE_DIR=/app/models/torchxrayvision
ENV XDG_CACHE_HOME=/app/.cache
ENV HOME=/app

# ============================================================================
# INSTALAR DEPENDENCIAS DEL SISTEMA
# ============================================================================
RUN apt-get update && apt-get install -y \
    # Dependencias para OpenCV y procesamiento de imágenes
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    # Dependencias para DICOM y imágenes médicas
    libfontconfig1 \
    libfreetype6 \
    libxft2 \
    # Herramientas del sistema
    curl \
    wget \
    git \
    # Dependencias para python-magic
    libmagic1 \
    # Herramientas de compilación
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================================
# CONFIGURAR DIRECTORIO DE TRABAJO
# ============================================================================
WORKDIR /app

# ============================================================================
# INSTALAR DEPENDENCIAS DE PYTHON
# ============================================================================
COPY requirements.txt .

# Actualizar pip e instalar dependencias
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# CREAR ESTRUCTURA DE DIRECTORIOS COMPLETA
# ============================================================================
RUN mkdir -p \
    /app/uploads \
    /app/models \
    /app/models/transformers \
    /app/models/huggingface \
    /app/models/torchxrayvision \
    /app/temp \
    /app/logs \
    /app/cache \
    /app/.cache \
    /app/.cache/torch \
    /app/.cache/torchxrayvision \
    /.torchxrayvision \
    /.torchxrayvision/models_data

# ============================================================================
# COPIAR CÓDIGO DE LA APLICACIÓN
# ============================================================================
COPY app/ ./app/
COPY .env .env
COPY README.md ./
COPY requirements.txt ./

# ============================================================================
# CONFIGURACIÓN DE SEGURIDAD Y PERMISOS
# ============================================================================
# Crear usuario no privilegiado
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser

# Configurar permisos (CRÍTICO para TorchXRayVision)
RUN chown -R appuser:appuser /app && \
    chown -R appuser:appuser /.torchxrayvision && \
    chmod -R 755 /app && \
    chmod -R 777 /app/models /app/uploads /app/logs /app/cache /app/temp && \
    chmod -R 777 /app/.cache /.torchxrayvision

# ============================================================================
# SCRIPT DE INICIO ROBUSTO
# ============================================================================
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    echo "🏥 Iniciando Radiology AI Backend v2.0..."\n\
    echo "======================================"\n\
    \n\
    # Configurar variables de entorno adicionales\n\
    export TORCHXRAYVISION_CACHE_DIR=/app/models/torchxrayvision\n\
    export XDG_CACHE_HOME=/app/.cache\n\
    export HOME=/app\n\
    \n\
    # Crear directorios si no existen (compatible con volúmenes)\n\
    echo "📁 Configurando directorios..."\n\
    mkdir -p /app/logs /app/models /app/uploads /app/temp /app/cache\n\
    mkdir -p /app/models/torchxrayvision\n\
    mkdir -p /app/.cache/torchxrayvision\n\
    mkdir -p /.torchxrayvision/models_data 2>/dev/null || true\n\
    \n\
    # Configurar permisos (solo si es posible)\n\
    echo "🔧 Configurando permisos..."\n\
    chmod 777 /app/models /app/uploads /app/temp /app/cache /app/logs 2>/dev/null || echo "⚠️ No se pudieron configurar algunos permisos (normal en sistemas de archivos montados)"\n\
    chmod 777 /app/models/torchxrayvision 2>/dev/null || true\n\
    chmod 777 /.torchxrayvision 2>/dev/null || true\n\
    \n\
    echo "💻 Sistema preparado:"\n\
    echo "   • TorchXRayVision Cache: $TORCHXRAYVISION_CACHE_DIR"\n\
    echo "   • Home Directory: $HOME"\n\
    echo "   • Cache Directory: $XDG_CACHE_HOME"\n\
    echo "   • Puerto: $PORT"\n\
    \n\
    echo "🚀 Iniciando servidor en puerto $PORT..."\n\
    exec uvicorn app.main:app --host $HOST --port $PORT --workers 1\n\
    ' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# ============================================================================
# CONFIGURACIÓN DE PUERTOS Y SALUD
# ============================================================================
EXPOSE 8002

HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# ============================================================================
# CAMBIAR A USUARIO NO PRIVILEGIADO
# ============================================================================
USER appuser

# ============================================================================
# COMANDO DE INICIO
# ============================================================================
CMD ["/app/entrypoint.sh"]