# Compendio de Slides y Scripts

## Slide 1: Visión General de la Herramienta

### Bullets
- **Diagnóstico automatizado de documentos de proyecto** — Analiza PRODOC, informes de evaluación y otros documentos contra rúbricas institucionales de la OIT, generando puntuaciones (1-5) con evidencia trazable extraída directamente del texto.
- **Múltiples dimensiones de análisis** — Evalúa calidad preliminar, integración de género, transición justa, sostenibilidad y metodologías participativas, permitiendo seleccionar criterios específicos según el tipo de documento.
- **Resultados exportables y trazables** — Genera reportes en Excel con análisis narrativo, puntuación y citas textuales que respaldan cada evaluación, facilitando la rendición de cuentas y la mejora de propuestas.

### Script
#### Bullet 1 – Diagnóstico automatizado de documentos de proyecto
La herramienta permite cargar documentos en formato Word (.docx) —como Documentos de Proyecto, informes de progreso o evaluaciones— y analizarlos automáticamente contra matrices de criterios de la OIT. El sistema extrae el texto completo, lo envía a un modelo de lenguaje (GPT-4) junto con los descriptores de los niveles 1-5 y obtiene: evidencia relevante, comparación con los descriptores, puntuación justificada y citas textuales. **Limitaciones**: únicamente .docx, dependiente de la calidad del documento, posibles interpretaciones incorrectas y tiempos de procesamiento que varían según la cantidad de criterios.

#### Bullet 2 – Múltiples dimensiones de análisis
La aplicación ofrece cinco módulos principales (Valoración Preliminar, Atributos específicos, Sostenibilidad, Chat con Documentos y Estadísticas de Recomendaciones). Cada rúbrica contiene indicadores con descriptores de desempeño. Por ejemplo, para "Integración del Enfoque de Género" el modelo identifica el nivel que mejor corresponde al texto. **Limitaciones**: las puntuaciones son orientativas, pueden variar entre ejecuciones y no sustituyen la validación humana ni evalúan la implementación real.

#### Bullet 3 – Resultados exportables y trazables
Cada evaluación genera un Excel con: Dimensión, Criterio, Indicador, Score, Análisis, Evidencia y Errores. La trazabilidad proviene de las citas textuales que respaldan cada puntuación, lo que permite verificar interpretaciones, detectar áreas de mejora y documentar decisiones. **Limitaciones**: evidencias extensas pueden truncarse, el análisis narrativo refleja la interpretación del modelo y los resultados deben descargarse antes de cerrar la sesión.

---

## Slide 2: ¿Cómo funciona la herramienta?

### Bullets
- **Extracción estructurada del documento** — El sistema convierte el archivo .docx en texto plano, preservando la estructura y contenido completo para su análisis.
- **Evaluación criterio por criterio con IA** — Un modelo de lenguaje (GPT-4) recibe cada indicador de la rúbrica junto con sus descriptores de nivel (1-5) y el texto del documento, generando puntuación, análisis y evidencia textual.
- **Procesamiento paralelo y trazabilidad** — Múltiples criterios se evalúan simultáneamente para reducir tiempos, y cada resultado incluye la cita exacta del documento que respalda la puntuación asignada.

### Script
#### Bullet 1 – Extracción estructurada del documento
Al cargarse un .docx, la app crea una copia temporal, extrae el texto con `docx2python`, obtiene métricas (tamaño y palabras) y elimina el archivo temporal. El texto extraído se convierte en el contexto único que verá la IA; no se almacena permanentemente.

#### Bullet 2 – Evaluación criterio por criterio con IA
Para cada criterio se arma un prompt que incluye: instrucciones (“Eres un evaluador experto…”), el indicador, los descriptores nivel 1-5, el texto completo y la instrucción de respuesta en JSON. Ejemplo de salida:
```json
{
  "score": 3,
  "analysis": "El documento menciona la importancia de considerar las necesidades diferenciadas de hombres y mujeres...",
  "evidence": "\"El proyecto beneficiará a trabajadores y trabajadoras...\" (pág. 12)"
}
```
El modelo no accede a internet ni a fuentes externas.

#### Bullet 3 – Procesamiento paralelo y trazabilidad
Se envían hasta 48 criterios en paralelo para reducir tiempos (1-6 minutos según volumen). Tras recibir todas las respuestas, se ordenan por dimensión, se generan visualizaciones y se habilitan descargas. Cada puntuación conserva la cita textual que la respalda. **Limitaciones clave**: ventana de contexto (documentos muy largos pueden truncarse), variabilidad (±1 punto), interpretación literal y costo por uso. Diagrama resumido:
```
Usuario ─▶ Servidor (extrae texto) ─▶ OpenAI (evalúa)
   ▲                                   │
   └───────── Resultados + Excel ◀────┘
```

---

## Slide 3: La herramienta como asistente, no como juez

### Bullets
- **Acelera, no reemplaza** — La IA realiza una primera lectura sistemática; el juicio profesional sigue siendo indispensable.
- **Detecta oportunidades, no defectos** — Las puntuaciones bajas señalan áreas de mejora, no críticas personales.
- **Insumo para el diálogo, no veredicto final** — Los resultados son un punto de partida para la discusión técnica.

### Script
#### Bullet 1 – Acelera, no reemplaza
Funciona como un corrector avanzado: revisa sistemáticamente decenas de criterios y extrae evidencia, pero no decide. El valor humano sigue siendo clave para interpretar contexto, restricciones y decisiones estratégicas.

#### Bullet 2 – Detecta oportunidades, no defectos
Una puntuación baja indica que el documento no explicita cierto tema, no que el equipo haya fallado. Puede deberse a que la información está en anexos, aún no se desarrolla o se decidió omitir.

#### Bullet 3 – Insumo para el diálogo, no veredicto final
Use los resultados para abrir conversaciones: “¿Queremos que este tema quede más explícito?” Las puntuaciones orientan, pero la decisión de ajustar o no recae en el equipo evaluador.

---

## Slide 4: ¿Qué significa realmente una puntuación baja?

### Bullets
- **No mide calidad del trabajo, mide presencia en el documento**
- **El documento no es el proyecto**
- **Oportunidad de mejora antes del escrutinio externo**

### Script
#### Bullet 1 – No mide calidad del trabajo
La herramienta evalúa lo que está escrito. Si el análisis de género está en un documento separado, la puntuación será baja aunque el trabajo real sea excelente.

#### Bullet 2 – El documento no es el proyecto
Un PRODOC impecable puede describir un proyecto débil y viceversa. La IA no conoce la realidad del terreno; evalúa el texto disponible.

#### Bullet 3 – Oportunidad de mejora
Es mejor detectar vacíos antes de enviar el documento. Las puntuaciones bajas permiten reforzar secciones clave y anticipar observaciones externas.

---

## Slide 5: El rol del especialista es más importante que nunca

### Bullets
- **Contexto que la IA no tiene**
- **Validación crítica**
- **Decisión final siempre humana**

### Script
#### Bullet 1 – Contexto que la IA no tiene
El modelo ignora negociaciones, restricciones, acuerdos con mandantes o fases previas. Solo el equipo sabe por qué el documento dice lo que dice.

#### Bullet 2 – Validación crítica
La IA puede equivocarse: confundir términos, variar puntuaciones o no reconocer jerga técnica. Su revisión es el control de calidad imprescindible.

#### Bullet 3 – Decisión final siempre humana
Ningún resultado debe usarse de forma automática para aprobar o rechazar. Las decisiones sobre ajustes, prioridades y uso de tiempo son exclusivamente humanas.

---

## Slide 6: Uso recomendado de la herramienta

### Bullets
- **Antes de enviar, no después de rechazar**
- **Dialogue con los resultados**
- **Documente sus decisiones**

### Script
#### Bullet 1 – Antes de enviar
Integre la herramienta como paso de auto-revisión: Borrador → Diagnóstico IA → Revisión en equipo → Ajustes → Envío.

#### Bullet 2 – Dialogue con los resultados
Preguntas útiles: “¿Aplica este criterio? ¿Falta información o el modelo no la encontró? ¿Qué priorizamos dado el contexto?”

#### Bullet 3 – Documente sus decisiones
Si decide no ajustar algo señalado por la IA, registre el motivo. Esto fortalece la trazabilidad y demuestra análisis consciente.
