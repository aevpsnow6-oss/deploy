# 🚀 Quick Start - ILO Document Evaluator

Guía rápida de 5 minutos para comenzar a usar la aplicación.

---

## ⚡ Inicio Rápido (3 pasos)

### Windows

1. **Crear archivo `.env`** con tu clave de OpenAI:
   ```
   OPENAI_API_KEY=tu-clave-aqui
   ```

2. **Doble clic en `start.bat`**

3. **¡Listo!** La aplicación se abre en tu navegador

### Mac / Linux

1. **Crear archivo `.env`** con tu clave de OpenAI:
   ```
   OPENAI_API_KEY=tu-clave-aqui
   ```

2. **Ejecutar en terminal:**
   ```bash
   ./start.sh
   ```

3. **¡Listo!** La aplicación se abre en tu navegador

---

## 🔑 Obtener tu Clave de OpenAI

1. Ir a: https://platform.openai.com/api-keys
2. Iniciar sesión o crear cuenta
3. Click en "Create new secret key"
4. Copiar la clave (empieza con `sk-proj-...`)
5. Pegar en archivo `.env`

**Nota:** Necesitarás créditos en tu cuenta OpenAI. Comprar créditos en: https://platform.openai.com/settings/organization/billing

---

## 📝 Uso Básico

1. **Subir documento**: Click en "Browse files" y seleccionar archivo .docx
2. **Click "Analizar documento"**: Esperar mientras se procesa
3. **Seleccionar rúbricas**: Marcar las rúbricas que quieres aplicar
4. **Seleccionar criterios**: Elegir criterios específicos (o "Seleccionar todos")
5. **Click "Procesar y Evaluar"**: El análisis comienza
6. **Ver resultados**: Tabla con scores, análisis y evidencia
7. **Descargar**: Click en "Download" para obtener Excel

---

## 💰 Estimación de Costos

**Costo aproximado por evaluación:**
- Documento pequeño (10-20 criterios): $0.10 - $0.30 USD
- Documento mediano (30-50 criterios): $0.50 - $1.00 USD
- Documento grande (60+ criterios): $1.50 - $3.00 USD

**Factores que afectan el costo:**
- Número de criterios evaluados
- Longitud del documento
- Modelo usado (gpt-5-mini es más económico)

---

## ⏱️ Tiempo de Procesamiento

**Tiempos aproximados:**
- Procesamiento inicial del documento: 30-60 segundos
- Por cada criterio: 15-30 segundos
- 20 criterios ≈ 6-10 minutos total
- 50 criterios ≈ 15-25 minutos total

**Nota:** La evaluación se hace en paralelo (3 criterios a la vez) para ser más rápida.

---

## 🐛 Problemas Comunes

### "Python no encontrado"
**Solución:** Instalar Python 3.10+ desde python.org

### "No module named 'streamlit'"
**Solución:** Los scripts instalan automáticamente. Si falla, ejecutar manualmente:
```bash
pip install -r requirements.txt
```

### "Port 8501 already in use"
**Solución:** Cerrar otras instancias de Streamlit o cambiar puerto:
```bash
streamlit run oli_v6_deploy.py --server.port=8502
```

### La aplicación se detiene a mitad de evaluación
**Solución:** Esto fue un bug que ya está corregido. Asegúrate de tener la última versión.

### Errores de API 429 (Rate Limit)
**Solución:** La aplicación reintenta automáticamente. Si persiste, esperar 1-2 minutos.

---

## 📞 ¿Necesitas Ayuda?

1. **README completo:** Ver `README.md` para documentación detallada
2. **Distribución:** Ver `DISTRIBUCION.md` para info de deployment
3. **Soporte:** [Especificar canal de soporte]

---

## 🎓 Tips Pro

1. **Evaluar por rúbricas:** Para documentos grandes, evaluar una rúbrica a la vez en lugar de todas juntas

2. **Reutilizar documentos procesados:** Si ya procesaste un documento y quieres evaluar más criterios, no es necesario subirlo de nuevo (la app guarda el documento procesado en sesión)

3. **Revisar la columna "Error":** Si un criterio tiene score 0, revisar la columna Error para ver qué pasó

4. **Guardar resultados inmediatamente:** Descargar los resultados en Excel antes de refrescar la página

5. **Uso local vs cloud:** Ejecutar localmente elimina restricciones de memoria y timeouts

---

## 🔄 Actualizaciones

Para actualizar a una nueva versión:
1. Descargar nueva versión
2. Reemplazar archivos (mantener tu `.env`)
3. Ejecutar `start.bat` o `start.sh` normalmente

---

¡Disfruta usando ILO Document Evaluator! 🎉
