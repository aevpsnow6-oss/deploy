# ILO Document Evaluator - Local Deployment Guide

Sistema de evaluación de documentos con criterios de la OIT usando GPT-5 y análisis avanzado.

## 📋 Requisitos Previos

### Opción 1: Instalación Directa (Recomendado para desarrollo)
- **Python 3.10 o superior**
- **pip** (gestor de paquetes de Python)
- **Clave API de OpenAI** (obtén una en: https://platform.openai.com/api-keys)

### Opción 2: Docker (Recomendado para distribución)
- **Docker Desktop** (descarga desde: https://www.docker.com/products/docker-desktop)
- **Clave API de OpenAI**

---

## 🚀 Instalación y Uso

### Método 1: Ejecución Directa (Desarrollo)

#### Windows
1. Descargar o clonar el repositorio
2. Crear archivo `.env` con tu clave de API:
   ```
   OPENAI_API_KEY=tu-clave-api-aqui
   ```
3. Hacer doble clic en `start.bat`
4. La aplicación se abrirá automáticamente en tu navegador en http://localhost:8501

#### Mac / Linux
1. Descargar o clonar el repositorio
2. Crear archivo `.env` con tu clave de API:
   ```
   OPENAI_API_KEY=tu-clave-api-aqui
   ```
3. Abrir terminal en la carpeta del proyecto
4. Ejecutar: `./start.sh`
5. La aplicación se abrirá automáticamente en tu navegador en http://localhost:8501

**Nota:** Los scripts `start.bat` y `start.sh` automáticamente:
- Crean un entorno virtual de Python
- Instalan las dependencias necesarias
- Inician la aplicación
- Abren el navegador

### Método 2: Docker (Producción)

#### Windows
1. Asegurarte que Docker Desktop esté instalado y ejecutándose
2. Crear archivo `.env` con tu clave de API
3. Hacer doble clic en `start-docker.bat`
4. Esperar a que se construya la imagen (solo la primera vez)
5. La aplicación se abrirá en http://localhost:8501

#### Mac / Linux
1. Asegurarte que Docker Desktop esté instalado y ejecutándose
2. Crear archivo `.env` con tu clave de API
3. Abrir terminal en la carpeta del proyecto
4. Ejecutar: `./start-docker.sh`
5. Esperar a que se construya la imagen (solo la primera vez)
6. La aplicación se abrirá en http://localhost:8501

**Comandos útiles de Docker:**
```bash
# Ver logs en tiempo real
docker-compose logs -f

# Detener la aplicación
docker-compose down

# Reiniciar la aplicación
docker-compose restart

# Reconstruir la imagen (después de cambios)
docker-compose up -d --build
```

---

## 📂 Estructura de Archivos

```
ilo-evaluator/
├── oli_v6_deploy.py          # Aplicación principal
├── Rubricas_6ago2025.xlsx    # Archivo de rúbricas
├── requirements.txt          # Dependencias Python
├── .env                      # Configuración (API key) - CREAR ESTE ARCHIVO
├── .env.example              # Plantilla para .env
│
├── Dockerfile                # Configuración Docker
├── docker-compose.yml        # Orquestación Docker
├── .dockerignore            # Archivos ignorados por Docker
│
├── start.bat                 # Launcher Windows (directo)
├── start.sh                  # Launcher Mac/Linux (directo)
├── start-docker.bat          # Launcher Windows (Docker)
├── start-docker.sh           # Launcher Mac/Linux (Docker)
│
├── data/                     # Carpeta para datos (creada automáticamente)
└── README.md                 # Esta guía
```

---

## 🔧 Configuración Avanzada

### Variables de Entorno (.env)

```bash
# OpenAI API Key (Requerido)
OPENAI_API_KEY=sk-proj-...

# Opcional: Configuración del modelo
OPENAI_MODEL=gpt-5-mini
```

### Modificar Puerto de la Aplicación

**Método directo:** Editar el comando en `start.bat` o `start.sh`:
```bash
streamlit run oli_v6_deploy.py --server.port=8502
```

**Método Docker:** Editar `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Cambiar 8502 al puerto deseado
```

---

## 🐛 Solución de Problemas

### "Python no está instalado o no está en PATH"
- **Windows:** Instalar desde python.org, marcar "Add Python to PATH"
- **Mac:** `brew install python@3.10`
- **Linux:** `sudo apt install python3.10 python3-pip`

### "Docker is not installed or not running"
- Instalar Docker Desktop desde docker.com
- Asegurarse de que Docker Desktop esté ejecutándose (ver ícono en la bandeja)

### ".env file not found"
- Copiar `.env.example` a `.env`
- Agregar tu clave de OpenAI API en el archivo `.env`

### "Port 8501 is already in use"
- Detener otras instancias de Streamlit o aplicaciones en el puerto 8501
- O cambiar el puerto (ver Configuración Avanzada)

### La aplicación se congela durante la evaluación
- **Esto es normal para documentos grandes**
- El progreso se muestra en la barra
- Para documentos muy grandes (>50 criterios), considerar evaluar por rúbricas separadas

### Errores de API (429 Rate Limit)
- La aplicación tiene reintentos automáticos
- Si persiste, considerar:
  - Reducir el número de criterios por ejecución
  - Esperar unos minutos y volver a intentar
  - Verificar límites de tu cuenta OpenAI

---

## 📦 Distribución a Usuarios Finales

### Opción A: Paquete ZIP (Simple)
1. Comprimir toda la carpeta en un archivo .zip
2. Incluir instrucciones:
   - Extraer el .zip
   - Crear archivo .env con API key
   - Ejecutar start.bat (Windows) o start.sh (Mac/Linux)

### Opción B: Docker Image (Profesional)
1. Construir la imagen:
   ```bash
   docker build -t ilo-evaluator:1.0 .
   ```
2. Guardar la imagen:
   ```bash
   docker save ilo-evaluator:1.0 -o ilo-evaluator.tar
   ```
3. Distribuir el archivo .tar + instrucciones:
   ```bash
   # Usuario carga la imagen
   docker load -i ilo-evaluator.tar

   # Usuario ejecuta
   docker run -p 8501:8501 -e OPENAI_API_KEY=su-key ilo-evaluator:1.0
   ```

### Opción C: Instalador Ejecutable (Avanzado)
Usar PyInstaller para crear un .exe o .app standalone:
```bash
pip install pyinstaller
pyinstaller --onefile --windowed launcher.py
```

---

## 📊 Uso de la Aplicación

1. **Subir documento:** Cargar archivo .docx para evaluación
2. **Seleccionar rúbricas:** Elegir las rúbricas a aplicar
3. **Seleccionar criterios:** Marcar criterios específicos dentro de cada rúbrica
4. **Procesar y Evaluar:** Click en el botón para iniciar análisis
5. **Revisar resultados:** Ver tabla con análisis, scores y evidencia
6. **Descargar:** Obtener resultados en formato Excel (ZIP)

---

## 📝 Notas Importantes

- **Costos:** Cada evaluación consume créditos de OpenAI (modelo gpt-5-mini)
- **Privacidad:** La aplicación NO almacena documentos ni resultados
- **Conexión:** Requiere internet para acceder a la API de OpenAI
- **Memoria:** Ejecutar localmente elimina restricciones de memoria cloud
- **Rendimiento:** 3 evaluaciones simultáneas (MAX_WORKERS=3) para evitar rate limiting

---

## 🆘 Soporte

Para problemas técnicos o preguntas:
1. Revisar la sección "Solución de Problemas" arriba
2. Verificar logs:
   - **Método directo:** Ver terminal/consola
   - **Docker:** `docker-compose logs -f`
3. Verificar estado de OpenAI API: https://status.openai.com

---

## 📄 Licencia

[Especificar licencia del proyecto]

---

## 🔄 Actualizaciones

Para actualizar a una nueva versión:

**Método directo:**
```bash
git pull  # Si usas git
# O descargar nueva versión y reemplazar archivos
```

**Docker:**
```bash
docker-compose down
docker-compose up -d --build
```
