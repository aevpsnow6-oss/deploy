# Guion · Bloque 4 · "Cómo funciona la tecnología y cuáles son sus límites"
**Ahmed Eid · Día 1, jueves 27 ago 2026 · 9:30–10:30 (Lima) · 30 min + preguntas integradas**

> **Reparto del tiempo.** 30 min de exposición + 13 min de preguntas que la agenda integra
> a este bloque = **43 min reales**. El guion abajo suma **28 min**, dejando 2 min de holgura
> y los 13 de preguntas. Si va retrasado, los recortes seguros están marcados con ✂.

| Parte | Diapositivas | Tiempo | Acumulado |
|---|---|---|---|
| Apertura | 1–2 | 1:30 | 1:30 |
| A · De dónde viene y qué es | 3–5 | 4:00 | 5:30 |
| B · Cómo está construido | 6–9 | 6:00 | 11:30 |
| C · Por qué repite | 10–13 | 5:30 | 17:00 |
| D · Límites y comparación | 14–16 | 3:30 | 20:30 |
| E · Demostración | 17–23 | 6:30 | 27:00 |
| Cierre | 24–27 | 1:00 | 28:00 |

---

## Apertura · diapositivas 1–2 · 1:30

**[Dp. 1]**
"Buenos días. En los bloques anteriores vieron *qué* hace el Agente y *quién* debería usarlo.
Mi parte es la de abajo del capó: **cómo funciona, por qué a veces cambia de opinión, y qué
no puede hacer**. Al final vamos a verlo funcionando."

**[Dp. 2]** Recorrer los cinco puntos en voz alta, sin detenerse.
"Un aviso: las preguntas están integradas a mi bloque. **No las guarden para el final** —
si algo no queda claro, interrúmpanme, porque probablemente le pase a alguien más."

---

## Parte A · De dónde viene y qué es · diapositivas 3–5 · 4:00

**[Dp. 3 · De Streamlit al Agente] — 1:30**
"Esto no nació como un GPT. Nació como una aplicación Streamlit que ya existía y que algunos
de ustedes vieron. Funcionaba, pero tenía dos barreras: había que instalarla y había que
aprender su interfaz."

Señalar la última fila: "**El motor es el mismo código.** Lo que cambió es la puerta de
entrada. Esto importa porque significa que no estamos ante una herramienta nueva sin
rodaje: es la misma lógica de evaluación, servida de otra forma."

**[Dp. 4 · Qué es un GPT] — 1:45**
"Voy a quitar el misterio. Un modelo de lenguaje **predice texto**. Ha leído cantidades
enormes de texto y aprendió qué palabras tienden a seguir a otras."

Ir punto por punto. Detenerse en el tercero:
"No es un buscador. No hay una lista de respuestas correctas guardada en algún lado que el
sistema recupere. Cada vez que evalúa, **razona en ese momento** sobre el documento que
ustedes le dieron."

Cerrar con la banda roja — **es la frase más importante de mi bloque**:
"El Agente solo ve lo que está escrito en el PRODOC. Lo que el especialista sabe y no
documentó, para el Agente **no existe**. Vamos a volver a esto varias veces."

**[Dp. 5 · Lo que no hace] — 0:45**
Leer solo la columna de la izquierda, rápido. "Cinco cosas que no hace. Me detengo en una:
no da siempre la misma respuesta. Eso suena a defecto y en un minuto les explico por qué es
inevitable y qué hicimos al respecto."

---

## Parte B · Cómo está construido · diapositivas 6–9 · 6:00

**[Dp. 6 · Cómo se incorporó la rúbrica] — 1:30**
"El Appraisal Checklist no se le 'pegó' al modelo como un archivo adjunto. Se **digitalizó
criterio por criterio**."

Recorrer los cinco pasos. Enfatizar el 3:
"Cada criterio tiene una **regla de decisión explícita**, escrita. No es el modelo decidiendo
libremente qué merece un Yes."

**[Dp. 7 · Las 5 secciones] — 1:00**
"76 criterios, 5 secciones. La sección 3, marco de resultados, es la más pesada con 27."

Banda: "**Pueden filtrar.** Y les recomiendo hacerlo. Evaluar solo la sección 3 tarda menos,
cuesta menos y es más fácil de revisar que 76 criterios de golpe."

**[Dp. 8 · De criterio a tests] — 2:30** ← *núcleo conceptual del bloque*
"Este es el corazón de cómo funciona. Tomemos un criterio real: el 1.5.6, enfoque
transformador en materia de género."

"El Agente **no** se pregunta '¿este PRODOC tiene buen enfoque de género?'. Eso sería una
impresión. Se pregunta tres cosas cerradas, una por una." — leer T1, T2, T3.

"Y luego aplica una fórmula." — señalar la línea DECISIÓN.
"Los tres se cumplen: Yes. Solo algunos: Partial. Ninguno de los dos anclajes: No."

Cerrar: "Por eso el resultado es **auditable**. Cuando ustedes vean un 'Partial' en este
criterio, pueden ir al Excel y ver exactamente qué test falló y con qué cita del documento."

**[Dp. 9 · DEDICADO vs MARCO] — 1:00**
"Un problema clásico: un PRODOC menciona 'género' catorce veces y no hace nada al respecto."

"El Agente clasifica cada mención antes de contarla. Si el tema aparece en una lista de
grupos, o en lenguaje genérico de inclusión, eso es **MARCO** y no cuenta como evidencia.
Cuenta si hay un producto, un indicador desagregado, una actividad o una partida
presupuestaria."

Banda roja: "Si toda la evidencia es marco, el resultado **debe** ser No, sin importar
cuántas veces aparezca la palabra."

---

## Parte C · Por qué repite · diapositivas 10–13 · 5:30

**[Dp. 10 · Por qué 10 veces] — 1:30**
"Aquí está la respuesta a '¿por qué puede cambiar una respuesta?'."

"Un modelo de lenguaje **no es determinista**. La misma pregunta sobre el mismo documento
puede dar respuestas distintas. Eso no es un error de programación: es cómo funciona."

"Preguntar una sola vez sería entregarles el resultado de **un solo lanzamiento**. Así que
cada criterio se evalúa **10 veces** de forma independiente y se toma el resultado más
frecuente."

Banda: "76 criterios por 10 corridas son unas **760 consultas** al modelo. Por eso tarda
minutos y no segundos, y por eso insisto en filtrar."

**[Dp. 11 · De dónde viene la aleatoriedad] — 1:00** ✂ *(recortable a 30 s: leer solo la
primera fila y la banda)*
"Cuatro fuentes. La primera es inherente al modelo; las otras tres las mitigamos con diseño."

**[Dp. 12 · Estabilidad] — 2:00** ← *el concepto que más van a usar*
"De esas 10 corridas, ¿cuántas coincidieron? Ese porcentaje es la **estabilidad**, y va en
el Excel, criterio por criterio."

Recorrer la tabla. Detenerse en la fila roja del medio:
"Si ven 60%, significa que de diez corridas, seis dijeron una cosa y cuatro otra. Eso **no**
es un resultado que puedan usar tal cual. Es una señal de que ese criterio necesita su
juicio."

Cerrar con la banda — **decirlo despacio**:
"La estabilidad **no mide si el Agente acertó**. Mide cuánta confianza interna tuvo. Un 100%
equivocado es perfectamente posible. Por eso siempre, siempre, se verifica la evidencia."

**[Dp. 13 · Subjetividad] — 1:00**
"Además de la estabilidad, cada criterio trae una etiqueta de subjetividad."

"Alta subjetividad significa que el juicio depende del contexto institucional, no solo del
texto. Si un enfoque de género es 'transformador' es discutible entre dos expertos; si el
PRODOC tiene un marco lógico es verificable."

"Dos cosas disparan la marca de revisión humana: **subjetividad alta**, o **estabilidad bajo
80%**. Esa columna es su cola de trabajo."

---

## Parte D · Límites y comparación · diapositivas 14–16 · 3:30

**[Dp. 14 · Ventajas y desventajas] — 1:15**
Leer en pares: "Cobertura: revisa los 76 sin cansarse — pero revisa lo escrito, no lo sabido."

Banda: "El costo depende del tamaño del documento y de cuántos criterios evalúen. **Filtrar
por sección es la palanca directa de ahorro.**"

**[Dp. 15 · GPT Empresarial] — 1:15** ← *pregunta garantizada del público*
"Alguien va a preguntar: '¿y si uso el GPT empresarial de la OIT y le subo la rúbrica?'"

Recorrer la tabla rápido. Detenerse en dos filas: repetición y estabilidad.
"El GPT empresarial hace **una sola pasada** y no les dice nada sobre su propia consistencia."

Fila roja: "Y hoy, además, no hay acceso a este Agente desde el entorno empresarial."

Banda: "El valor agregado no es 'usar IA'. Es la rúbrica institucional aplicada criterio por
criterio, repetida y medida."

**[Dp. 16 · Tres límites] — 1:00**
Leer los tres, sin adornos. Cerrar: "Ninguna salida es una determinación oficial de la OIT."

---

## Parte E · Demostración · diapositivas 17–23 · 6:30

> **⚠ Preparación obligatoria.** Tener el Excel de una corrida previa **ya descargado y
> abierto**. Si hace una corrida en vivo, lance **una sola subsección** y siga hablando
> mientras corre. Nunca deje la sala mirando una barra de progreso.

**[Dp. 17 · portadilla] — 0:15** "Vamos a verlo."

**[Dp. 18 · El flujo] — 0:45** Recorrer los cinco pasos.
Banda: "Digan el alcance en el mismo mensaje en que suben el archivo."

**[Dp. 19 · Pasos 1–2] — 1:00**
"Si abren el Agente y le dicen 'hola', se presenta solo y les explica qué puede filtrar.
Eso no consume una evaluación." — leer los tres ejemplos de instrucción.

**[Dp. 20 · Paso 3] — 0:45**
"Mientras esperan, esto es lo que ocurre." — recorrer.
Banda: "Si es la primera del día puede tardar más en arrancar: el servicio se suspende por
inactividad. Es esperado, no es una falla."

**[Dp. 21 · El Excel] — 1:30**
"Una sola hoja, una fila por criterio. Cinco grupos de columnas." — recorrer.
Banda: "**'Not Found' no es 'No'.** 'No' dice que el criterio no se cumple. 'Not Found' dice
que el documento no permite determinarlo. La acción que sigue es distinta: uno es rediseño,
el otro es documentación."

**[Dp. 22 · Cómo leer una fila] — 1:15**
"En este orden, siempre." — recorrer los cinco pasos.
Banda: "Nunca acepten una respuesta sin abrir la evidencia."

**[Dp. 23 · Localizar evidencia] — 1:00**
"La cita textual está en el Excel. Cópienla y búsquenla en el PRODOC con Ctrl+F."

Los tres desenlaces — **es el puente al bloque del día 2**:
"Si la información sí estaba: descartan el diagnóstico. Si el formulador la conoce pero no
está escrita: se mejora el PRODOC. Si no existe: se resuelve antes de cerrar la formulación."

Banda: "¿Existe? ¿Es suficiente? ¿Está documentado?"

---

## Cierre · diapositivas 24–27 · 1:00

**[Dp. 24 · Vinculación] — 0:30** ✂ *(recortable si va justo de tiempo)*
"El Excel no es el final. Pueden seguir conversando con el Agente sobre el resultado, y hay
agentes específicos para género, participación, transición justa y sostenibilidad."

**[Dp. 26 · Qué recordar] — 0:30**
"Cuatro cosas." — leerlas.
Cerrar: "El valor de la herramienta depende de la calidad de la revisión humana posterior.
La herramienta les ahorra la lectura mecánica para que ustedes gasten su tiempo donde
aporta: decidiendo."

**[Dp. 27]** "Preguntas."

---

## Respuestas de bolsillo (13 min de preguntas)

**¿Por qué puede cambiar una respuesta?**
El modelo no es determinista. Por eso corremos cada criterio 10 veces y reportamos cuánto
coincidieron. Si les preocupa la variabilidad, la estabilidad es exactamente la métrica que
la hace visible.

**¿Qué significa estabilidad de 50–70%?**
Que las corridas no se pusieron de acuerdo. Casi siempre indica un criterio ambiguo o
evidencia insuficiente en el documento. Es una invitación a mirarlo ustedes, no un resultado
para usar tal cual.

**¿Por qué no encontró algo que yo sé que existe?**
Dos posibilidades: o no está escrito donde ustedes creen, o el Agente no lo reconoció.
Verifiquen con la evidencia citada. Si estaba y no lo vio, descarten ese diagnóstico — y
avísennos, porque es información útil para ajustar la rúbrica.

**¿Qué significa que un criterio sea subjetivo?**
Que dos expertos podrían discrepar legítimamente. El Agente les dedica más razonamiento y
los marca para revisión. No significa que el resultado esté mal; significa que su juicio
pesa más ahí.

**¿Por qué no me da un informe con recomendaciones de mejora?**
Porque no incorpora conocimiento fuera del documento. Una recomendación necesita saber del
país, del mandante, del historial — cosas que el Agente no tiene. Además, no queremos
restringir las posibilidades de interacción: pueden pedirle eso al chat, sabiendo de dónde
sale.

**¿Cuánto cuesta evaluar un PRODOC y de qué depende?**
Depende del tamaño del documento y de cuántos criterios evalúen. Una corrida completa son
unas 760 consultas; una sección es una fracción de eso. Filtrar es la forma directa de
controlar el costo.

**¿Se pueden evaluar otros documentos?**
Sí, con otros agentes: atributos específicos (género, participación, transición justa) y
sostenibilidad, que aplica según la etapa del ciclo.

**Si uso el GPT Empresarial, ¿obtengo los mismos resultados?**
No. Una sola pasada, sin estabilidad, sin la rúbrica descompuesta en tests, y sin Excel
estructurado. Y hoy no hay acceso a este Agente desde el entorno empresarial.

**¿Cuándo lo puedo usar? / ¿Cómo ayudo a institucionalizarlo?**
→ *Derivar a Cybele (bloque 3) y al día 2.*

**¿Está en línea con las IGDS? / ¿Cuánto cuesta el mantenimiento?**
→ *Derivar: son decisiones institucionales, no técnicas.*
