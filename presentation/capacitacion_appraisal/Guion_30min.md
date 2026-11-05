# Guion · Bloque 4 · "Cómo funciona la tecnología y cuáles son sus límites"
**Ahmed Eid · Día 1, jueves 27 ago 2026 · 9:30–10:30 (Lima) · 30 min + preguntas integradas**

> **Reparto del tiempo.** 30 min de exposición + 13 min de preguntas que la agenda integra
> a este bloque = **43 min reales**. El guion abajo suma **31:45**: pasarse un poco está
> previsto y aceptado. Los ✂ quedan por si el bloque anterior se alarga. Los números de
> diapositiva son los del deck de 35 (29 de contenido + 6 portadillas).

| Parte | Diapositivas | Tiempo | Acumulado |
|---|---|---|---|
| Apertura | 1–2 | 1:30 | 1:30 |
| A · De dónde viene y qué es | 3–7 | 5:00 | 6:30 |
| B · Cómo está construido | 8–13 | 7:15 | 13:45 |
| C · Por qué repite | 14–18 | 5:30 | 19:15 |
| D · Límites y comparación | 19–22 | 3:30 | 22:45 |
| E · Demostración | 23–30 | 8:00 | 30:45 |
| Cierre | 31–35 | 1:00 | 31:45 |

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

## Parte A · De dónde viene y qué es · diapositivas 3–7 · 5:00

**[Dp. 4 · De Streamlit al Agente] — 1:30**
"Esto no nació como un GPT. Nació como una aplicación Streamlit que ya existía y que algunos
de ustedes vieron. Funcionaba, pero tenía dos barreras: había que instalarla y había que
aprender su interfaz."

Señalar la última fila: "**El motor es el mismo código.** Lo que cambió es la puerta de
entrada. Esto importa porque significa que no estamos ante una herramienta nueva sin
rodaje: es la misma lógica de evaluación, servida de otra forma."

**[Dp. 5 · Qué era la aplicación Streamlit] — 1:00** ← *contexto: de dónde sale todo esto*
"Antes de que existiera el Agente, esto era una aplicación con seis pestañas."

Recorrer la tabla: "Las tres primeras son las que hoy son Agentes: valoración de calidad,
atributos específicos y sostenibilidad. Las otras dos —el chat documental y la clasificación
de recomendaciones— siguen viviendo en la aplicación."

"¿Por qué se construyó? Porque el Checklist se aplicaba a mano, criterio por criterio, y eso
consumía horas de especialista con una cobertura de evaluación limitada."

**[Dp. 6 · Qué es un GPT] — 1:45**
"Voy a quitar el misterio. Un modelo de lenguaje **predice texto**. Ha leído cantidades
enormes de texto y aprendió qué palabras tienden a seguir a otras."

Ir punto por punto. Detenerse en el tercero:
"No es un buscador. No hay una lista de respuestas correctas guardada en algún lado que el
sistema recupere. Cada vez que evalúa, **razona en ese momento** sobre el documento que
ustedes le dieron."

Cerrar con la banda roja — **es la frase más importante de mi bloque**:
"El Agente solo ve lo que está escrito en el PRODOC. Lo que el especialista sabe y no
documentó, para el Agente **no existe**. Vamos a volver a esto varias veces."

**[Dp. 7 · Lo que no hace] — 0:45**
Leer solo la columna de la izquierda, rápido. "Cinco cosas que no hace. Me detengo en una:
no da siempre la misma respuesta. Eso suena a defecto y en un minuto les explico por qué es
inevitable y qué hicimos al respecto."

---

## Parte B · Cómo está construido · diapositivas 8–13 · 7:15

**[Dp. 9 · Cómo se incorporó la rúbrica] — 1:30**
"El Appraisal Checklist no se le 'pegó' al modelo como un archivo adjunto. Se **digitalizó
criterio por criterio**."

Recorrer los cinco pasos. Enfatizar el 3:
"Cada criterio tiene una **regla de decisión explícita**, escrita. No es el modelo decidiendo
libremente qué merece un Yes."

**[Dp. 10 · Las 5 secciones] — 1:00**
"76 criterios, 5 secciones. La sección 3, marco de resultados, es la más pesada con 27."

Banda: "**Pueden filtrar.** Y les recomiendo hacerlo. Por sección, por subsección, o por
criterios sueltos: si un revisor les señaló el 1.1.1 y el 3.1.1, evalúan sólo esos dos.
Tarda menos, cuesta menos y es más fácil de revisar que 76 criterios de golpe."

**[Dp. 11 · De criterio a tests] — 2:30** ← *núcleo conceptual del bloque*
"Este es el corazón de cómo funciona. Tomemos un criterio real: el 1.5.6, enfoque
transformador en materia de género."

"El Agente **no** se pregunta '¿este PRODOC tiene buen enfoque de género?'. Eso sería una
impresión. Se pregunta tres cosas cerradas, una por una." — leer T1, T2, T3.

"Y luego aplica una fórmula." — señalar la línea DECISIÓN.
"Los tres se cumplen: Yes. Solo algunos: Partial. Ninguno de los dos anclajes: No."

Cerrar: "Por eso el resultado es **auditable**. Cuando ustedes vean un 'Partial' en este
criterio, pueden ir al Excel y ver exactamente qué test falló y con qué cita del documento."

**[Dp. 12 · DEDICADO vs MARCO] — 1:00**
"Un problema clásico: un PRODOC menciona 'género' catorce veces y no hace nada al respecto."

"El Agente clasifica cada mención antes de contarla. Si el tema aparece en una lista de
grupos, o en lenguaje genérico de inclusión, eso es **MARCO** y no cuenta como evidencia.
Cuenta si hay un producto, un indicador desagregado, una actividad o una partida
presupuestaria."

Banda roja: "Si toda la evidencia es marco, el resultado **debe** ser No, sin importar
cuántas veces aparezca la palabra."

**[Dp. 13 · Un caso completo] — 1:15** ← *la diapositiva que ancla todo lo anterior*
"Hasta aquí les he explicado el mecanismo. Veámoslo funcionando sobre un texto."

Izquierda: "Tres extractos de un PRODOC. Ilustrativos, no es un documento real."

Derecha, uno por uno: "El primer chequeo se cumple: la sección 2.3 nombra y distingue los tres
enfoques. El segundo no: declara el enfoque pero nunca explica el mecanismo. El tercero sí: la
actividad 2.4 revisa las reglas de acceso al crédito, y tiene presupuesto propio."

Señalar la cuarta tarjeta — **es el punto que más se discute**:
"Fíjense en la sección 3.1: menciona explícitamente a cuatro grupos, incluidas las mujeres.
Y aun así **no cuenta como evidencia**, porque es una lista sin seguimiento. Eso es marco, no
dedicado. Aquí es donde muchos dirán «pero si sí lo mencionamos»."

Cerrar: "Se requieren los tres. Se cumplen dos. Resultado: Partial. Y todo el camino desde el
texto hasta el veredicto es verificable."

---

## Parte C · Por qué repite · diapositivas 14–18 · 5:30

**[Dp. 15 · Por qué 10 veces] — 1:30**
"Aquí está la respuesta a '¿por qué puede cambiar una respuesta?'."

"Un modelo de lenguaje **no es determinista**. La misma pregunta sobre el mismo documento
puede dar respuestas distintas. Eso no es un error de programación: es cómo funciona."

"Preguntar una sola vez sería entregarles el resultado de **un solo lanzamiento**. Así que
cada criterio se evalúa **10 veces** de forma independiente y se toma el resultado más
frecuente."

Banda: "76 criterios por 10 corridas son unas **760 consultas** al modelo. Cinco criterios
concretos son 50. Por eso tarda minutos y no segundos, y por eso insisto en filtrar."

**[Dp. 16 · De dónde viene la aleatoriedad] — 1:00** ✂ *(recortable a 30 s: leer solo la
primera fila y la banda)*
"Cuatro fuentes. La primera es inherente al modelo; las otras tres las mitigamos con diseño."

**[Dp. 17 · Estabilidad] — 2:00** ← *el concepto que más van a usar*
"De esas 10 corridas, ¿cuántas coincidieron? Ese porcentaje es la **estabilidad**, y va en
el Excel, criterio por criterio."

Recorrer la tabla. Detenerse en la fila roja del medio:
"Si ven 60%, significa que de diez corridas, seis dijeron una cosa y cuatro otra. Eso **no**
es un resultado que puedan usar tal cual. Es una señal de que ese criterio necesita su
juicio."

Cerrar con la banda — **decirlo despacio**:
"La estabilidad **no mide si el Agente acertó**. Mide cuánto coincidieron las corridas. Y un
100% equivocado existe: lo vimos en el criterio 1.1.1. Un PRODOC enlazaba sus resultados con
el P&B diciendo «aligned with», alineado con. El criterio pide *contribución*, no alineación.
Las diez corridas leyeron igual de mal: diez de diez, «Yes». Un error sistemático es estable.
Por eso siempre, siempre, se verifica la evidencia. Ese criterio ya se corrigió en la rúbrica."

**[Dp. 18 · Subjetividad] — 1:00**
"Además de la estabilidad, cada criterio trae una etiqueta de subjetividad."

"Alta subjetividad significa que el juicio depende del contexto institucional, no solo del
texto. Si un enfoque de género es 'transformador' es discutible entre dos expertos; si el
PRODOC tiene un marco lógico es verificable."

"Y aquí una distinción que importa. La hoja **«Revisión prioritaria»** del Excel es lo que
hay que revisar **sí o sí**: los criterios inestables, bajo 80%, y los que salieron «No». En
un PRODOC completo son unos 17 a 20, uno de cada cuatro. No es una lista de excepciones: es
la parte del trabajo que necesita su juicio."

"La alta subjetividad **no entra** en esa hoja. Queda marcada en la columna de revisión
recomendada, y la revisan según su criterio. El orden: primero la hoja prioritaria, después
la subjetividad."

---

## Parte D · Límites y comparación · diapositivas 19–22 · 3:30

**[Dp. 20 · Ventajas y desventajas] — 1:15**
Leer en pares: "Cobertura: revisa los 76 sin cansarse — pero revisa lo escrito, no lo sabido."

Banda: "El costo depende del tamaño del documento y de cuántos criterios evalúen. Completo,
unas 760 consultas; cinco criterios concretos, 50. **Filtrar es la palanca directa de ahorro.**"

**[Dp. 21 · GPT Empresarial] — 1:15** ← *pregunta garantizada del público*
"Alguien va a preguntar: '¿y si uso el GPT empresarial de la OIT y le subo la rúbrica?'"

Recorrer la tabla rápido. Detenerse en dos filas: repetición y estabilidad.
"El GPT empresarial hace **una sola pasada** y no les dice nada sobre su propia consistencia."

Fila roja: "Y hoy, además, no hay acceso a este Agente desde el entorno empresarial."

Banda: "El valor agregado no es 'usar IA'. Es la rúbrica institucional aplicada criterio por
criterio, repetida y medida."

**[Dp. 22 · Tres límites] — 1:00**
Leer los tres, sin adornos. Cerrar: "Ninguna salida es una determinación oficial de la OIT."

---

## Parte E · Demostración · diapositivas 23–30 · 8:00

> **⚠ Preparación obligatoria.** Tener el Excel de una corrida previa **ya descargado y
> abierto**. Si hace una corrida en vivo, lance **dos o tres criterios sueltos** (p. ej.
> «solo 1.1.1 y 3.1.1»: 20 consultas, termina en un par de minutos) y siga hablando mientras
> corre. Recuerde que el Agente **no informa solo**: escriba «estado» para mostrar el avance.

**[Dp. 23 · portadilla] — 0:15** "Vamos a verlo."

**[Dp. 24 · El flujo] — 0:45** Recorrer los cinco pasos.
Banda: "Digan el alcance en el mismo mensaje en que suben el archivo."

**[Dp. 25 · Pasos 1–2] — 1:00**
"Si abren el Agente y le dicen 'hola', se presenta solo y les explica qué puede filtrar.
Eso no consume una evaluación." — leer los tres ejemplos de instrucción. En el tercero:
"criterios sueltos, y el informe en inglés si lo necesitan. Los veredictos son los mismos en
los dos idiomas."

**[Dp. 26 · Paso 3] — 0:45**
"Mientras esperan, esto es lo que ocurre." — recorrer. En el paso 2: "Al lanzar, el Agente
les confirma el alcance: tantos criterios, tantas consultas. **Compruébenlo.** Si pidieron
cinco y dice 76, algo falló: no sigan."
Banda: "El Agente **no informa solo**. Escriban «estado» cuando quieran, y responde «Evaluando:
180 de 760 consultas (24%). Quedan unos 3 minutos». Cuando termine, esa consulta les entrega el
archivo. Y la primera evaluación del día puede tardar más en arrancar, porque el servicio se
suspende por inactividad."

**[Dp. 27 · El Excel] — 1:45**
"Tres hojas. La primera, «Resultado Diagnostico»: una fila por criterio, cinco grupos de
columnas." — recorrer la tabla.

"La segunda es **por donde empiezan**: «Revisión prioritaria». Los inestables y los «No», cada
uno con su motivo. Es la revisión obligatoria."

"La tercera, «Rubrica aplicada»: la definición de cada criterio que se evaluó. Con eso el
archivo se puede auditar solo, sin abrir la rúbrica por separado."
Banda: "**'Not Found' no es 'No'.** 'No' dice que el criterio no se cumple. 'Not Found' dice
que el documento no permite determinarlo. La acción que sigue es distinta: uno es rediseño,
el otro es documentación."

**[Dp. 28 · Cómo leer una fila] — 1:15**
"En este orden, siempre." — recorrer los cinco pasos.
Banda: "Nunca acepten una respuesta sin abrir la evidencia."

**[Dp. 29 · Qué contiene el Razonamiento] — 1:15** ← *la diapositiva que más preguntas evita*
"Esta es la celda que más miedo daba, y la hemos reescrito entera."

Recorrer las cuatro partes señalando el bloque de la izquierda:
"Arriba, **POR QUÉ**: el motivo en una línea. Si van con prisa, lean solo eso — «se cumplen
2 de 3 chequeos; falta articular cómo cuestiona normas de poder»."

"Debajo, **VERIFICACIÓN**: cada chequeo escrito completo, con su marca y su justificación.
Fíjense en que ya no dice «T2: falso» — dice la pregunta entera. No hay que ir a buscar qué
era T2."

"Después **ESTABILIDAD**, que ya conocen, y al final **REGLA**: la regla formal del Checklist,
para quien necesite trazar el resultado hasta la rúbrica."

Cerrar: "El análisis es exactamente el mismo de antes: los mismos chequeos, la misma evidencia,
la misma regla. Lo único que cambió es que ahora se lee sin descifrar nada."

**[Dp. 30 · Localizar evidencia] — 1:00**
"La cita textual está en el Excel. Cópienla y búsquenla en el PRODOC con Ctrl+F."

Los tres desenlaces — **es el puente al bloque del día 2**:
"Si la información sí estaba: descartan el diagnóstico. Si el formulador la conoce pero no
está escrita: se mejora el PRODOC. Si no existe: se resuelve antes de cerrar la formulación."

Banda: "¿Existe? ¿Es suficiente? ¿Está documentado?"

---

## Cierre · diapositivas 31–35 · 1:00

**[Dp. 31 · Vinculación] — 0:30** ✂ *(recortable si va justo de tiempo)*
"El Excel no es el final. Pueden seguir conversando con el Agente sobre el resultado, y hay
agentes específicos para género, participación, transición justa y sostenibilidad."

**[Dp. 34 · Qué recordar] — 0:30**
"Cuatro cosas." — leerlas. En la última: "la hoja prioritaria es su punto de partida".
Cerrar: "El valor de la herramienta depende de la calidad de la revisión humana posterior.
La herramienta les ahorra la lectura mecánica para que ustedes gasten su tiempo donde
aporta: decidiendo."

**[Dp. 35]** "Preguntas."

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
los marca en la columna de revisión recomendada. No significa que el resultado esté mal;
significa que su juicio pesa más ahí. Ojo: no entran en la hoja «Revisión prioritaria», que
es sólo para inestables y «No».

**¿Por qué no me da un informe con recomendaciones de mejora?**
Porque no incorpora conocimiento fuera del documento. Una recomendación necesita saber del
país, del mandante, del historial — cosas que el Agente no tiene. Además, no queremos
restringir las posibilidades de interacción: pueden pedirle eso al chat, sabiendo de dónde
sale.

**¿Cuánto cuesta evaluar un PRODOC y de qué depende?**
Depende del tamaño del documento y de cuántos criterios evalúen. Una corrida completa son
unas 760 consultas; cinco criterios concretos, 50. Filtrar es la forma directa de controlar
el costo.

**¿Puedo evaluar sólo unos criterios concretos?**
Sí, en cualquier cantidad: «solo 1.1.1, 2.1.2 y 3.1.1». Escriban el código completo, de tres
niveles: «solo 1.1» evalúa la subsección entera. Si un código no existe, el Agente lo dice
antes de empezar.

**¿Puedo tener el informe en inglés?**
Sí. Escriban en inglés o pídanlo. El Excel sale entero en inglés, con los mismos veredictos:
la evaluación se hace siempre contra la rúbrica oficial en español.

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
