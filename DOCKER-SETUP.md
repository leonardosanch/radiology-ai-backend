# 🐳 Configuración Docker - Radiology AI Backend

## 🚀 Inicio Rápido

### 1. Setup Automático

```bash
# Ejecutar script de configuración
./setup-dev.sh

# Iniciar el sistema
docker-compose up --build -d
```

### 2. Verificar que funciona

```bash
# Ver logs
docker-compose logs -f radiology-ai-backend

# Probar API
curl http://localhost:8002/health
curl http://localhost:8002/ping

# Ver documentación
open http://localhost:8002/docs
```

## 🔧 Solución de Problemas

### TorchXRayVision no se descarga

El sistema está configurado para manejar automáticamente:

- ✅ Permisos de directorio cache
- ✅ Variables de entorno correctas
- ✅ Descarga automática de modelos

### Problemas de permisos

```bash
# En sistemas Linux/Mac
sudo chmod 777 uploads models logs temp cache

# O usar el script de setup
./setup-dev.sh
```

### Limpiar y reiniciar

```bash
# Parar todo
docker-compose down -v

# Limpiar imágenes
docker system prune -f

# Reconstruir
docker-compose up --build -d
```

## 📋 Variables de Entorno Importantes

```bash
# TorchXRayVision
TORCHXRAYVISION_CACHE_DIR=/app/models/torchxrayvision
XDG_CACHE_HOME=/app/.cache
HOME=/app

# API
HOST=0.0.0.0
PORT=8002
DEBUG=true
```

## 🏗️ Arquitectura

```
radiology-ai-backend/
├── app/                    # Código fuente
├── models/                 # Modelos de IA (persistente)
├── uploads/               # Imágenes subidas (persistente)
├── logs/                  # Logs del sistema (persistente)
├── cache/                 # Cache temporal (persistente)
├── temp/                  # Archivos temporales
├── Dockerfile             # ✅ Configuración optimizada
├── docker-compose.yml     # ✅ Servicios orquestados
└── setup-dev.sh          # ✅ Script de configuración
```

## 👥 Para Desarrolladores

### Configuración de desarrollo

```bash
# Variables de entorno para desarrollo local
echo "ENVIRONMENT=development" >> .env
echo "LOG_LEVEL=DEBUG" >> .env

# Reconstruir después de cambios
docker-compose up --build
```

### Debugging

```bash
# Entrar al contenedor
docker-compose exec radiology-ai-backend bash

# Ver variables de entorno
docker-compose exec radiology-ai-backend env | grep TORCH

# Verificar directorios
docker-compose exec radiology-ai-backend ls -la /app/models/
```

## 🔒 Seguridad

- ✅ Usuario no privilegiado (appuser)
- ✅ Permisos mínimos necesarios
- ✅ Variables de entorno seguras
- ✅ Health checks configurados

## 📊 Monitoreo

```bash
# Estado de contenedores
docker-compose ps

# Uso de recursos
docker stats

# Logs en tiempo real
docker-compose logs -f
```
