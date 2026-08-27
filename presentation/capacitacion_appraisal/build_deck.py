"""Spanish deck for the 30-min ILO training block (Ahmed).

Layout and palette live in ilo_deck.py, shared with build_deck_en.py.
"""

from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from ilo_deck import *  # noqa: F403 - motor de maquetación compartido
import ilo_deck

ilo_deck.set_style("sober")
prs = new_deck()

# ═══ 1 · PORTADA ═══════════════════════════════════════════════════════
s = blank_slide()
rect(s, 0, 0, Inches(0.42), EMU_H, BLUE)
rect(s, Inches(0.42), 0, Inches(0.09), EMU_H, RED)
tb, tf = textbox(s, Inches(1.25), Inches(1.65), Inches(11.3), Inches(2.4))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "Cómo funciona la tecnología\ny cuáles son sus límites"
r.font.size = Pt(50); r.font.bold = True; r.font.color.rgb = BLUE
p2 = tf.add_paragraph(); p2.space_before = Pt(18)
set_runs(p2, [N("Agente GPT · Appraisal Checklist de PRODOCs")], 27)
rect(s, Inches(1.25), Inches(4.6), Inches(3.1), Inches(0.07), RED)
tb3, tf3 = textbox(s, Inches(1.25), Inches(4.95), Inches(11.0), Inches(1.6))
for segs in [[B("Expositor:  "), N("Ahmed Eid")],
             [B("Sesión:  "), N("Día 1 · jueves 27 de agosto de 2026 · 9:30–10:30 (Lima)")],
             [B("Bloque:  "), N("Punto 4 · 30 minutos + preguntas integradas")]]:
    p = tf3.add_paragraph() if tf3.paragraphs[0].runs else tf3.paragraphs[0]
    p.space_after = Pt(8)
    set_runs(p, segs, 21)

# ═══ 2 · RUTA ══════════════════════════════════════════════════════════
s = slide_new("Qué vamos a cubrir")
steps(s, [
    ("1", "De dónde viene", "de la aplicación Streamlit al Agente GPT"),
    ("2", "Cómo está construido", "la rúbrica, los tests, la búsqueda de evidencia"),
    ("3", "Por qué repite", "aleatoriedad, 10 corridas y estabilidad"),
    ("4", "Qué NO hace", "límites reales y comparación con el GPT Empresarial"),
    ("5", "Demostración", "cargar → ejecutar → interpretar el Excel"),
], size=21)
band(s, [R("Las preguntas se integran a este bloque: "), N("interrúmpanme en cualquier momento.")])

divider("Parte 1", "De dónde viene y qué es", "De la aplicación Streamlit al Agente GPT · qué es realmente un modelo de lenguaje")

# ═══ 3 · DE STREAMLIT AL GPT ═══════════════════════════════════════════
s = slide_new("De dónde viene: de Streamlit al Agente GPT")
table(s, ["", "Aplicación Streamlit", "Agente GPT"],
      [[("Acceso", BLUE, True), "URL + contraseña de entorno", "Enlace de ChatGPT"],
       [("Instalación", BLUE, True), "Servidor propio, dependencias", "Ninguna: se abre en el navegador"],
       [("Curva de uso", BLUE, True), "Interfaz con pestañas y parámetros", "Conversación en lenguaje natural"],
       [("Rúbrica", BLUE, True), "Cargada en el servidor", "Cargada en el servidor (igual)"],
       [("Motor", BLUE, True), "El mismo código de evaluación", "El mismo código de evaluación"],
       [("Salida", BLUE, True), "Excel descargable", "Excel descargable (igual)"]],
      CL, CT, CW, [1.5, 3.2, 3.2], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("El motor no cambió. "), N("Cambió la puerta de entrada: el Agente elimina la barrera de instalación y de interfaz.")])

# ═══ 3b · LA APP STREAMLIT ═════════════════════════════════════════════
s = slide_new("Qué era la aplicación Streamlit")
bullets(s, [
    ([B("Por qué se construyó."), N("  El Appraisal Checklist se aplicaba a mano, criterio por criterio. "
      "Revisar un PRODOC completo consumía horas de especialista y la cobertura de evaluación era limitada.")], 0, None),
], t=CT, h=Inches(1.05), size=19, gap=6)
table(s, ["Pestaña", "Qué hacía", "Sigue disponible"],
      [[("1", BLUE, True), "Valoración preliminar de calidad de PRODOCs (Appraisal Checklist)", "Sí, ahora también como Agente"],
       [("2", BLUE, True), "Diagnóstico de atributos específicos: género, participación, transición justa", "Sí, como Agente propio"],
       [("3", BLUE, True), "Diagnóstico de sostenibilidad del proyecto", "Sí, como Agente propio"],
       [("4", BLUE, True), "Pregúntale a tus documentos: chat sobre uno o varios archivos", ("Solo en Streamlit", MUTED, False)],
       [("5 y 6", BLUE, True), "Clasificación de recomendaciones de evaluación (ES / EN)", ("Solo en Streamlit", MUTED, False)]],
      CL, Inches(2.42), CW, [1.1, 5.4, 2.4], fsize=17, header_fs=17, fill_to=Inches(6.22))
band(s, [B("Los tres primeros flujos son los que se convirtieron en Agentes.  "), N("El chat documental y la clasificación de recomendaciones siguen viviendo en la aplicación.")])

# ═══ 4 · QUÉ ES UN LLM ═════════════════════════════════════════════════
s = slide_new("Qué es un GPT, en términos sencillos")
bullets(s, [
    ([B("Un modelo de lenguaje predice texto."), N("  Ha leído enormes cantidades de texto y aprendió qué palabras siguen a otras en un contexto dado.")], 0, BLUEMD),
    ([B("No es una base de datos."), N("  No «consulta» el PRODOC como quien busca en un archivo: lo lee completo y razona sobre él en ese momento.")], 0, BLUEMD),
    ([B("No es un buscador."), N("  No hay una lista de respuestas correctas guardada en algún lugar que el sistema recupere.")], 0, BLUEMD),
    ([B("Es un lector que sigue instrucciones."), N("  Le entregamos el documento, la rúbrica y una instrucción muy precisa, y devuelve un juicio con la cita que lo sustenta.")], 0, BLUEMD),
], size=22, gap=20)
band(s, [R("Consecuencia práctica:  "), N("el Agente solo puede ver lo que está escrito en el PRODOC. Lo que el especialista sabe y no documentó, no existe para él.")])

# ═══ 5 · LO QUE NO HACE ════════════════════════════════════════════════
s = slide_new("Lo que el Agente no hace")
table(s, ["No hace esto", "Por qué", "Qué implica para usted"],
      [["Aprobar o rechazar un PRODOC", "No emite determinaciones oficiales de la OIT", "La decisión sigue siendo del appraiser"],
       ["Adivinar lo no documentado", "Solo lee el texto entregado", "Un «No» puede ser una brecha documental, no de diseño"],
       ["Recordar evaluaciones anteriores", "Cada corrida parte de cero", "No compara automáticamente entre proyectos"],
       ["Redactar el PRODOC corregido", "No es su propósito ni su alcance", "Le señala dónde mirar, no qué escribir"],
       ["Dar siempre la misma respuesta", "Hay aleatoriedad inherente", "Por eso repetimos y medimos estabilidad"]],
      CL, CT, CW, [2.4, 2.9, 3.1], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("Diseño deliberado:  "), N("preferimos un diagnóstico acotado y auditable antes que una recomendación amplia sin respaldo en el documento.")])

divider("Parte 2", "Cómo está construido", "La rúbrica digitalizada, los chequeos y la búsqueda de evidencia")

# ═══ 6 · CÓMO SE INCORPORÓ LA RÚBRICA ══════════════════════════════════
s = slide_new("Cómo se incorporó el Appraisal Checklist")
steps(s, [
    ("1", "Se digitalizó la rúbrica", "cada criterio pasó a una fila estructurada, no a un texto libre"),
    ("2", "Se descompuso en tests", "cada criterio se expresa como preguntas booleanas T1, T2, T3…"),
    ("3", "Se escribió la regla de decisión", "una fórmula explícita determina Yes / Partial / No"),
    ("4", "Se fijaron anclas verificables", "términos y expresiones que el Agente debe buscar en el texto"),
    ("5", "Se marcó la subjetividad", "cada criterio se etiqueta Alta / Media / Baja"),
], size=20)
band(s, [B("La rúbrica vive en el servidor. "), N("El usuario nunca la sube: se evalúa siempre contra la misma versión institucional.")])

# ═══ 7 · LAS 5 SECCIONES ═══════════════════════════════════════════════
s = slide_new("Qué evalúa: 76 criterios en 5 secciones")
table(s, ["Sección", "Contenido", "Criterios", "Subsecciones"],
      [[("1", BLUE, True), "Pertinencia", ("20", RED, True), "1.1 – 1.5"],
       [("2", BLUE, True), "Validez del diseño", ("13", RED, True), "2.1 – 2.4"],
       [("3", BLUE, True), "Marco de resultados y R&M", ("27", RED, True), "3.1 – 3.7"],
       [("4", BLUE, True), "Implementación", ("14", RED, True), "4.1 – 4.4"],
       [("5", BLUE, True), "Presentación", ("2", RED, True), "5.1 – 5.2"]],
      CL, CT, CW, [1.1, 4.0, 1.2, 1.9], fsize=20, header_fs=19, fill_to=Inches(6.22))
band(s, [B("Se puede filtrar:  "), N("rúbrica completa, una sección («solo la sección 3») o subsecciones concretas («solo 1.1 y 2.3»). Filtrar reduce tiempo y costo.")])

# ═══ 8 · ANATOMÍA DE UN CRITERIO ═══════════════════════════════════════
s = slide_new("De criterio a tests: cómo «entiende» un criterio")
rect(s, CL, CT, Inches(0.05), Inches(1.30), BLUEMD)
tbc, tfc = textbox(s, CL + Inches(0.22), CT, CW - Inches(0.3), Inches(1.30))
pc = tfc.paragraphs[0]
set_runs(pc, [B("Criterio 1.5.6"), N("   ·   Sección 1 Pertinencia   ·   subjetividad "), R("Alta")], 15)
pc2 = tfc.add_paragraph(); pc2.space_before = Pt(4)
set_runs(pc2, [I("«Siempre que es posible, la propuesta promueve y destaca el uso de un enfoque "
                 "transformador en materia de género.»")], 19)
pc3 = tfc.add_paragraph(); pc3.space_before = Pt(6)
set_runs(pc3, [N("Así redactado no es evaluable: se descompone en preguntas cerradas.")], 15)
table(s, ["Test", "Pregunta cerrada", "Respuesta"],
      [[("T1", BLUE, True), "¿Distingue el tipo de enfoque (sensible / responsivo / transformador)?", ("sí / no", GRAY, True)],
       [("T2", BLUE, True), "¿Articula cómo el proyecto cuestiona normas o relaciones de poder?", ("sí / no", GRAY, True)],
       [("T3", BLUE, True), "¿Hay acciones dedicadas a transformar relaciones, no solo a incluir mujeres?", ("sí / no", GRAY, True)]],
      CL, Inches(2.78), CW, [0.75, 6.2, 1.15], fsize=20, header_fs=19, fill_to=Inches(4.95))
rect(s, CL, Inches(5.20), CW, Inches(0.78), TINT)
rect(s, CL, Inches(5.20), Inches(0.05), Inches(0.78), BLUEMD)
tb, tf = textbox(s, CL + Inches(0.22), Inches(5.20), CW - Inches(0.3), Inches(0.78), MSO_ANCHOR.MIDDLE)
p = tf.paragraphs[0]
set_runs(p, [B("DECISIÓN:   "), N("T1 ∧ T2 ∧ T3 → "), ("Yes", RED, True), N("      ·      T1 ∨ T3 (sin los tres) → "), ("Partial", RED, True), N("      ·      ¬T1 ∧ ¬T3 → "), ("No", RED, True)], 17)
band(s, [B("Esto es lo que hace auditable el resultado:  "), N("usted puede revisar test por test dónde y por qué el Agente llegó a esa conclusión.")])

# ═══ 9 · CÓMO BUSCA EVIDENCIA ══════════════════════════════════════════
s = slide_new("Cómo busca evidencia: DEDICADO vs MARCO")
bullets(s, [
    ([B("No basta con que una palabra aparezca en el documento."), N("  El Agente clasifica cada mención antes de contarla como evidencia.")], 0, None),
], t=CT, h=Inches(0.9), size=21, gap=6)
table(s, ["", "MARCO  (no cuenta)", "DEDICADO  (sí cuenta)"],
      [[("Qué es", BLUE, True), "El tema aparece mencionado, sin desarrollo propio", "El tema tiene un espacio propio en el diseño"],
       [("Ejemplos", BLUE, True), "Listas de ≥3 grupos · lenguaje de inclusión genérico · enumeraciones «entre otros»", "Producto o resultado que lo nombra · indicador desagregado · actividad dedicada · partida presupuestaria · meta cuantificable"]],
      CL, Inches(2.15), CW, [1.3, 3.3, 3.6], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [R("Regla dura:  "), N("si toda la evidencia citable es MARCO, el resultado debe ser «No» o «Not Found», sin importar cuántas veces se nombre el tema.")])

# ═══ 9b · CASO COMPLETO ════════════════════════════════════════════════
s = slide_new("Un caso completo: del texto del PRODOC al veredicto")

LX, LW = CL, Inches(5.85)
RX, RW = CL + LW + Inches(0.38), CW - LW - Inches(0.38)
TY, PH = Inches(1.26), Inches(4.72)

# ── izquierda · lo que dice el documento ──────────────────────────────
rect(s, LX, TY, LW, Inches(0.42), BLUE)
tbh, tfh = textbox(s, LX + Inches(0.14), TY, LW - Inches(0.2), Inches(0.42), MSO_ANCHOR.MIDDLE)
set_runs(tfh.paragraphs[0], [WB("LO QUE DICE EL PRODOC")], 15)
rect(s, LX, TY + Inches(0.42), LW, PH - Inches(0.42), GRAYLT)
tbd, tfd = textbox(s, LX + Inches(0.18), TY + Inches(0.54), LW - Inches(0.36), PH - Inches(0.66))
EXTRACTOS = [
    [B("§ 2.3 Enfoque de género")],
    [I("«El proyecto adopta un enfoque de género transformador, distinguiéndolo de los enfoques meramente sensibles o responsivos.»")],
    [N("")],
    [B("§ 3.1 Beneficiarios")],
    [I("«Se priorizará la participación de mujeres, jóvenes, personas con discapacidad y pueblos indígenas.»")],
    [N("")],
    [B("§ 4.2 Actividad 2.4")],
    [I("«Mesas de trabajo con cooperativas para revisar los criterios de acceso al crédito que excluyen a las mujeres titulares. Presupuesto: USD 18.000.»")],
]
first = True
for segs in EXTRACTOS:
    par = tfd.paragraphs[0] if first else tfd.add_paragraph()
    first = False
    par.space_after = Pt(3)
    set_runs(par, segs, 14)

# ── derecha · cómo lo evalúa ──────────────────────────────────────────
rect(s, RX, TY, RW, Inches(0.42), BLUE)
tbh2, tfh2 = textbox(s, RX + Inches(0.14), TY, RW - Inches(0.2), Inches(0.42), MSO_ANCHOR.MIDDLE)
set_runs(tfh2.paragraphs[0], [WB("CÓMO LO EVALÚA EL AGENTE")], 15)

CHEQUEOS = [
    ("✓", BLUEMD, "T1 · Distingue el tipo de enfoque", "§ 2.3 nombra y distingue los tres enfoques."),
    ("✗", RED, "T2 · Articula cómo cuestiona normas de poder", "declara el enfoque, pero no explica el mecanismo en ninguna sección."),
    ("✓", BLUEMD, "T3 · Acciones dedicadas a transformar relaciones", "§ 4.2 revisa reglas de acceso al crédito, con presupuesto propio."),
    ("–", GRAY, "§ 3.1 no cuenta como evidencia", "lista de cuatro grupos sin seguimiento: es MARCO, no DEDICADO."),
]
top = TY + Inches(0.52)
alto = Inches(1.02)
grupo = []
for marca, color, titulo, detalle in CHEQUEOS:
    rect(s, RX, top, RW, alto, GRAYLT if marca != "–" else WHITE)
    rect(s, RX, top, Inches(0.05), alto, color)
    tbm, tfm = textbox(s, RX + Inches(0.12), top, Inches(0.5), alto, MSO_ANCHOR.MIDDLE)
    pm = tfm.paragraphs[0]; pm.alignment = PP_ALIGN.CENTER
    set_runs(pm, [(marca, color, True)], 20)
    tbc, tfc = textbox(s, RX + Inches(0.62), top, RW - Inches(0.78), alto, MSO_ANCHOR.MIDDLE)
    pc = tfc.paragraphs[0]
    set_runs(pc, [B(titulo)], 13.5)
    pc2 = tfc.add_paragraph(); pc2.space_before = Pt(1)
    set_runs(pc2, [N(detalle)], 12.5)
    grupo.append(tbc)
    top = top + alto + Inches(0.11)
register_group(grupo)

band(s, [B("DECISIÓN:  "), N("se requieren T1 ∧ T2 ∧ T3. Se cumplen T1 y T3, falta T2  →  "), R("PARTIAL"), N("   ·   extracto ilustrativo, no es un PRODOC real.")])

divider("Parte 3", "Por qué repite", "Aleatoriedad, diez corridas y qué significa la estabilidad")

# ═══ 10 · POR QUÉ REPITE ═══════════════════════════════════════════════
s = slide_new("Por qué cada criterio se evalúa 10 veces")
bullets(s, [
    ([B("Un modelo de lenguaje no es determinista."), N("  La misma pregunta sobre el mismo documento puede producir respuestas distintas.")], 0, BLUEMD),
    ([B("Preguntar una sola vez sería frágil."), N("  Estaríamos entregando el resultado de un único lanzamiento.")], 0, BLUEMD),
    ([B("Solución: repetir y consolidar."), N("  Cada criterio se evalúa 10 veces de forma independiente y se toma el resultado más frecuente (la moda).")], 0, BLUEMD),
    ([B("El desacuerdo es información, no ruido."), N("  Cuántas de las 10 corridas coincidieron es un dato que se reporta y que usted debe leer.")], 0, BLUEMD),
], size=22, gap=20)
band(s, [B("Una valoración completa "), N("son 76 criterios × 10 corridas ≈ "), R("760 consultas al modelo"), N(". Por eso tarda y por eso conviene filtrar por sección.")])

# ═══ 11 · DE DÓNDE VIENE LA ALEATORIEDAD ═══════════════════════════════
s = slide_new("De dónde procede la aleatoriedad")
table(s, ["Fuente", "Qué ocurre", "Cómo lo controlamos"],
      [["Generación probabilística", "El modelo elige entre continuaciones posibles; no siempre la misma", "10 corridas + resultado modal"],
       ["Criterios ambiguos", "Cuando la rúbrica admite lectura, el modelo puede inclinarse distinto", "Tests cerrados y reglas de decisión explícitas"],
       ["Evidencia dispersa", "Si la evidencia está repartida, distintas corridas citan pasajes distintos", "Filtro DEDICADO / MARCO"],
       ["Documentos extensos", "Más texto, más margen de lectura parcial", "Extracción estructurada y evaluación por criterio"]],
      CL, CT, CW, [2.3, 3.5, 2.6], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("No se elimina la aleatoriedad: se mide y se reporta. "), N("Eso es preferible a ocultarla detrás de una respuesta única.")])

# ═══ 12 · ESTABILIDAD ══════════════════════════════════════════════════
s = slide_new("Qué significa «estabilidad»")
tb, tf = textbox(s, CL, CT, CW, Inches(1.5))
p = tf.paragraphs[0]
set_runs(p, [B("Estabilidad (%)  =  "), N("de las 10 corridas, cuántas coincidieron con el resultado final.")], 26)
p2 = tf.add_paragraph(); p2.space_before = Pt(10)
set_runs(p2, [N("Umbral institucional: "), R("80%"), N(". Por debajo, el criterio se marca para revisión humana.")], 22)
table(s, ["Estabilidad", "Lectura", "Qué hacer"],
      [[("100 – 80%", BLUE, True), "El modelo fue consistente", "Tratar como diagnóstico sólido; verificar la evidencia"],
       [("79 – 50%", RED, True), "Hubo desacuerdo entre corridas", "Revisión humana obligatoria; leer el Resultado Alternativo"],
       [("< 50%", RED, True), "No hubo consenso real", "El criterio es ambiguo o la evidencia insuficiente: decida usted"]],
      CL, Inches(3.05), CW, [1.6, 2.8, 4.0], fsize=19, header_fs=18, fill_to=Inches(6.22))
band(s, [B("La estabilidad no mide si el Agente acertó. "), N("Mide cuánta confianza interna tuvo. Un 100% equivocado es posible: por eso siempre se verifica la evidencia.")])

# ═══ 13 · SUBJETIVIDAD ═════════════════════════════════════════════════
s = slide_new("Criterios subjetivos y revisión humana")
bullets(s, [
    ([B("Cada criterio trae una etiqueta de subjetividad: "), R("Alta · Media · Baja"), N(".")], 0, BLUEMD),
    ([B("Alta subjetividad"), N(" = el juicio depende del contexto institucional, no solo del texto. Ejemplo: si un enfoque de género es «transformador».")], 0, BLUEMD),
    ([B("Estos criterios reciben más razonamiento"), N(" del modelo y se marcan automáticamente en la columna «Revisión humana recomendada».")], 0, BLUEMD),
    ([B("Dos disparadores de esa marca:"), N("  subjetividad alta, o estabilidad por debajo de 80%.")], 0, BLUEMD),
], size=22, gap=18)
band(s, [R("Úsela como cola de trabajo:  "), N("empiece por los criterios marcados. Ahí es donde su juicio profesional aporta más valor.")])

divider("Parte 4", "Límites y comparación", "Qué cuesta, qué no hace, y en qué se diferencia del GPT Empresarial")

# ═══ 14 · VENTAJAS Y DESVENTAJAS ═══════════════════════════════════════
s = slide_new("Ventajas y desventajas de esta tecnología")
table(s, ["", "Ventaja", "Desventaja / costo"],
      [[("Cobertura", BLUE, True), "Revisa los 76 criterios sin fatiga ni sesgo de cansancio", "Revisa lo escrito, no lo sabido"],
       [("Tiempo", BLUE, True), "Libera horas del especialista para atender brechas", "Una corrida completa tarda varios minutos"],
       [("Trazabilidad", BLUE, True), "Cita la evidencia y muestra el razonamiento por test", "Genera mucho detalle: exige saber leerlo"],
       [("Consistencia", BLUE, True), "Aplica la misma rúbrica a todos los proyectos", "No es determinista: por eso repetimos"],
       [("Costo", BLUE, True), "Bajo por documento frente al tiempo humano equivalente", "Se paga por consulta: el presupuesto es finito"]],
      CL, CT, CW, [1.5, 3.5, 3.3], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("Tokens y presupuesto:  "), N("el costo depende del tamaño del PRODOC y de cuántos criterios se evalúen. Filtrar por sección es la palanca directa de ahorro.")])

# ═══ 15 · GPT EMPRESARIAL ══════════════════════════════════════════════
s = slide_new("¿Y si uso el GPT Empresarial de la OIT?")
table(s, ["", "GPT Empresarial", "Agente Appraisal Checklist"],
      [[("Rúbrica", BLUE, True), "Habría que subirla en cada conversación", "Cargada en el servidor, versión única"],
       [("Aplicación", BLUE, True), "Lectura general del archivo adjunto", "Cada criterio se ejecuta por separado, con sus tests"],
       [("Repetición", BLUE, True), "Una sola pasada", "10 corridas por criterio + resultado modal"],
       [("Estabilidad", BLUE, True), "No disponible", "Reportada por criterio, con umbral de 80%"],
       [("Salida", BLUE, True), "Texto en la conversación", "Excel estructurado, auditable y archivable"],
       [("Acceso hoy", BLUE, True), ("Sin acceso al Agente desde el entorno empresarial", RED, True), "Enlace de ChatGPT"]],
      CL, CT, CW, [1.5, 3.2, 3.6], fsize=17, header_fs=17, fill_to=Inches(6.22))
band(s, [B("El valor agregado no es «usar IA»:  "), N("es la rúbrica institucional aplicada criterio por criterio, repetida y medida. Eso el GPT genérico no lo reproduce.")])

# ═══ 16 · TRES LÍMITES ═════════════════════════════════════════════════
s = slide_new("Los tres límites que hay que tener presentes")
steps(s, [
    ("1", "Es una primera revisión completa,", "no un diagnóstico determinista. Señala brechas para que usted las examine."),
    ("2", "Trabaja sobre la evidencia del documento,", "no sobre el proyecto real ni sobre lo que usted sabe de él."),
    ("3", "No sustituye el juicio del appraiser.", "El resultado es un insumo; la determinación sigue siendo humana."),
], t=Inches(1.9), size=21)
band(s, [R("Ninguna salida constituye una determinación oficial de la OIT. "), N("Es una valoración asistida que requiere validación experta.")])

# ═══ PORTADILLA · LA HERRAMIENTA EN USO ═══════════════════════════════
divider("Parte 5", "La herramienta en uso", "Cargar · acotar · ejecutar · descargar · interpretar")

# ═══ 18 · EL FLUJO ═════════════════════════════════════════════════════
s = slide_new("El flujo completo, en cinco pasos")
steps(s, [
    ("1", "Cargar el PRODOC", "un único archivo .docx por evaluación"),
    ("2", "Seleccionar el alcance", "rúbrica completa, una sección o subsecciones concretas"),
    ("3", "Ejecutar", "el Agente lanza el trabajo y consulta su avance hasta terminar"),
    ("4", "Descargar el Excel", "es el registro auditable de la evaluación"),
    ("5", "Interpretar", "localizar evidencia y priorizar los criterios marcados"),
], size=21)
band(s, [B("Indique el alcance en el mismo mensaje en que sube el archivo:  "), N("ahorra una ronda de preguntas y acota el costo desde el inicio.")])

# ═══ 19 · PASO 1-2 ═════════════════════════════════════════════════════
s = slide_new("Pasos 1 y 2 · Cargar y acotar")
bullets(s, [
    ([B("Abra el Agente y salude."), N("  Se presenta solo: explica qué evalúa, qué secciones existen y qué puede filtrar. Eso no consume una evaluación.")], 0, BLUEMD),
    ([B("Adjunte un solo .docx."), N("  Si sube varios, el Agente le pedirá elegir uno: la evaluación es de un documento por vez.")], 0, BLUEMD),
    ([B("Diga qué quiere evaluar."), N("  Ejemplos de instrucción:")], 0, BLUEMD),
    ([I("«Evalúa este PRODOC con la rúbrica completa»")], 1, None),
    ([I("«Evalúa solo la sección 3 (Marco de resultados) y resume las brechas»")], 1, None),
    ([I("«Evalúa solo 1.1 y 2.3»")], 1, None),
], size=21, gap=13)
band(s, [R("Recomendación para empezar:  "), N("una sección primero. Llega antes, cuesta menos y permite calibrar la lectura antes de una corrida completa.")])

# ═══ 20 · PASO 3 ═══════════════════════════════════════════════════════
s = slide_new("Paso 3 · Qué ocurre mientras se ejecuta")
steps(s, [
    ("1", "El Agente descarga el documento", "y extrae su texto por secciones"),
    ("2", "Carga la rúbrica del servidor", "y filtra los criterios que usted pidió"),
    ("3", "Lanza las evaluaciones en paralelo", "cada criterio, 10 veces, hasta 48 consultas simultáneas"),
    ("4", "Consolida cada criterio", "resultado modal + porcentaje de estabilidad"),
    ("5", "Construye el Excel", "y lo adjunta a la conversación"),
], size=20)
band(s, [B("No hay que preguntar «¿ya terminó?»:  "), N("el Agente informa el avance solo — «180/760 (24%), quedan unos 3 minutos» — hasta entregar el archivo.")])

# ═══ 21 · EL EXCEL ═════════════════════════════════════════════════════
s = slide_new("Paso 4 · El Excel de resultados")
bullets(s, [
    ([B("Hoja 1 · «Resultado Diagnostico»"), N("  — una fila por criterio evaluado. Es el registro auditable: consérvelo.")], 0, None),
    ([B("Hoja 2 · «Rubrica aplicada»"), N("  — la definición de cada criterio evaluado: sus chequeos y sus reglas de decisión.")], 0, None),
], t=CT, h=Inches(1.15), size=20, gap=6)
table(s, ["Grupo de columnas", "Qué contiene", "Para qué sirve"],
      [[("Identificación", BLUE, True), "ID · Subsección · Criterio · Transversales", "Ubicar el criterio en el Checklist"],
       [("Resultado", BLUE, True), "Respuesta (Yes / Partial / No / Not Found / N/A)", "El diagnóstico del criterio"],
       [("Confianza", BLUE, True), "Estabilidad (%) · Estable (≥80%) · Resultado Alternativo", "Cuánto coincidieron las 10 corridas"],
       [("Sustento", BLUE, True), "Razonamiento (chequeo por chequeo) · Evidencia citada", "Verificar por qué llegó a ese resultado"],
       [("Prioridad", BLUE, True), ("Revisión humana recomendada", RED, True), "Su cola de trabajo"]],
      CL, Inches(2.42), CW, [2.0, 3.7, 2.6], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("La hoja 2 hace el archivo autocontenido:  "), N("quien audite ve la regla junto al veredicto, sin abrir la rúbrica por separado. Sólo incluye los criterios evaluados.")])

# ═══ 22 · ANATOMÍA DE UNA FILA ═════════════════════════════════════════
s = slide_new("Cómo leer una fila, en orden")
steps(s, [
    ("1", "Mire la Respuesta", "Yes / Partial / No / Not Found / N/A"),
    ("2", "Mire la Estabilidad", "¿coincidieron las 10 corridas o hubo desacuerdo?"),
    ("3", "Lea la Evidencia citada", "¿ese pasaje realmente sostiene el resultado?"),
    ("4", "Lea el Razonamiento", "qué chequeo falló y con qué justificación"),
    ("5", "Decida", "¿brecha de diseño, brecha documental, o error del Agente?"),
], size=21)
band(s, [R("«Not Found» no es «No»:  "), N("«No» afirma que el criterio no se cumple; «Not Found», que el documento no permite determinarlo. Nunca acepte una respuesta sin abrir la evidencia.")])

# ═══ 22b · ANATOMÍA DEL RAZONAMIENTO ═══════════════════════════════════
s = slide_new("Qué contiene la columna Razonamiento")

# bloque que reproduce la celda tal como se ve en el Excel
BX, BY, BW = CL, Inches(1.30), Inches(8.15)
rect(s, BX, BY, BW, Inches(4.92), GRAYLT)
rect(s, BX, BY, Inches(0.07), Inches(4.92), BLUEMD)
tbb, tfb = textbox(s, BX + Inches(0.22), BY + Inches(0.12), BW - Inches(0.42), Inches(4.7))
CELDA = [
    [B("POR QUÉ PARTIAL"), N(" · Se cumplen 2 de 3 chequeos.")],
    [N("Falta: articular cómo el proyecto cuestiona normas de poder.")],
    [N("")],
    [B("VERIFICACIÓN")],
    [(("✓ "), BLUEMD, True), N("¿Distingue el tipo de enfoque?")],
    [I("      la sección 3.2 distingue los tres enfoques.")],
    [(("✗ "), RED, True), N("¿Articula cómo cuestiona normas de poder?")],
    [I("      no se explica el mecanismo en ninguna sección.")],
    [(("✓ "), BLUEMD, True), N("¿Acciones dedicadas a transformar relaciones?")],
    [I("      actividad 2.4, con presupuesto asignado.")],
    [N("")],
    [B("ESTABILIDAD"), N(" · 6 de 10 corridas coincidieron. Alternativo: No.")],
    [B("REGLA"), N(" · se requieren los 3 chequeos  (T1 ∧ T2 ∧ T3)")],
]
first = True
for segs in CELDA:
    par = tfb.paragraphs[0] if first else tfb.add_paragraph()
    first = False
    par.space_after = Pt(3)
    set_runs(par, segs, 15)

# lectura de las cuatro partes, a la derecha
RX = CL + BW + Inches(0.30)
RW = CW - BW - Inches(0.30)
partes = [
    ("POR QUÉ", "el motivo, en una línea. Si va con prisa, lea solo esto."),
    ("VERIFICACIÓN", "cada chequeo enunciado completo, con ✓ / ✗ y su justificación."),
    ("ESTABILIDAD", "cuántas de las 10 corridas coincidieron."),
    ("REGLA", "la regla formal del Checklist, para auditoría."),
]
top = BY
grupo_partes = []
for titulo, detalle in partes:
    rect(s, RX, top, RW, Inches(1.16), WHITE)
    rect(s, RX, top, Inches(0.055), Inches(1.16), RED)
    tbp, tfp = textbox(s, RX + Inches(0.18), top, RW - Inches(0.3), Inches(1.16), MSO_ANCHOR.MIDDLE)
    pp = tfp.paragraphs[0]
    set_runs(pp, [B(titulo)], 17)
    pp2 = tfp.add_paragraph(); pp2.space_before = Pt(2)
    set_runs(pp2, [N(detalle)], 14)
    grupo_partes.append(tbp)
    top = top + Inches(1.25)
register_group(grupo_partes)

band(s, [B("Ya no hay que descifrar «T1 ∧ T2 ∧ T3»:  "), N("cada chequeo se lee solo. La regla formal sigue abajo, para quien necesite trazar el resultado.")])

# ═══ 23 · LOCALIZAR EVIDENCIA ══════════════════════════════════════════
s = slide_new("Paso 5 · Localizar la evidencia en el PRODOC")
bullets(s, [
    ([B("La columna Evidencia trae la cita textual"), N(" que el Agente usó. Búsquela en el PRODOC con Ctrl+F para verla en su contexto.")], 0, BLUEMD),
    ([B("Cuando la evidencia es una ausencia,"), N(" el Agente lo dice explícitamente: «No se encontró sección X». Eso también es un hallazgo verificable.")], 0, BLUEMD),
    ([B("Tres desenlaces posibles al verificar:")], 0, BLUEMD),
    ([R("La información sí está y el Agente no la reconoció "), N("→ se descarta el diagnóstico automático")], 1, None),
    ([R("La información no está pero el formulador la conoce "), N("→ mejorar el PRODOC")], 1, None),
    ([R("La información no existe aún "), N("→ resolver antes de cerrar la formulación, o trasladar a inception")], 1, None),
], size=21, gap=13)
band(s, [B("La fórmula:  "), N("¿existe?  →  ¿es suficiente?  →  ¿está documentado?")])

# ═══ 24 · VINCULACIÓN ══════════════════════════════════════════════════
s = slide_new("Vinculación con otros recursos")
table(s, ["Recurso", "Qué aporta", "Ejemplo de uso"],
      [[("Chatea con el Agente", BLUE, True), "Preguntar sobre el resultado ya generado, en lenguaje natural", "«Resume las brechas de la sección 3 en tres puntos»"],
       [("Otros GPTs desarrollados", BLUE, True), "Atributos Específicos (género, participación, transición justa) y Sostenibilidad", "Profundizar un tema que el Checklist solo toca de forma transversal"],
       [("Chatbot i-EVal", BLUE, True), "Evidencia de evaluaciones anteriores en la región", "«¿Qué problemas recurrentes han identificado las evaluaciones en proyectos con ministerios de trabajo?»"]],
      CL, CT, CW, [2.2, 3.4, 4.0], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("El Excel no es el final del proceso:  "), N("es el punto de partida de la conversación con el equipo formulador.")])

# ═══ 25 · PREGUNTAS ANTICIPADAS ════════════════════════════════════════
s = slide_new("Preguntas frecuentes")
table(s, ["Pregunta", "Respuesta breve"],
      [["¿Por qué puede cambiar una respuesta?", "El modelo no es determinista. Por eso repetimos 10 veces y reportamos estabilidad."],
       ["¿Qué significa una estabilidad de 50–70%?", "Las corridas no coincidieron: criterio ambiguo o evidencia insuficiente. Revisión humana."],
       ["¿Por qué no encontró algo que yo sé que existe?", "Solo lee el documento. Si está y no lo vio, verifíquelo y descarte el diagnóstico."],
       ["¿Por qué no me da recomendaciones de mejora?", "No incorpora conocimiento externo al documento. Las recomendaciones salen de su lectura."],
       ["¿Cuánto cuesta evaluar un PRODOC?", "Depende del tamaño del documento y de cuántos criterios se evalúen. Filtrar reduce el costo."],
       ["¿Puedo evaluar otros documentos?", "Para género, participación, transición justa y sostenibilidad existen agentes específicos."]],
      CL, CT, CW, [3.4, 6.0], fsize=17, header_fs=17, fill_to=Inches(7.05))

divider("Cierre", "Qué recordar", "")

# ═══ 26 · CIERRE ═══════════════════════════════════════════════════════
s = slide_new("Qué recordar de este bloque")
bullets(s, [
    ([B("El Agente lee el documento, no el proyecto."), N("  Un «No» puede ser una brecha de diseño o simplemente algo no documentado. Distinguirlo es trabajo suyo.")], 0, RED),
    ([B("Repite 10 veces y le dice cuánto coincidió."), N("  La estabilidad es una medida de confianza interna, no de acierto.")], 0, RED),
    ([B("Todo resultado viene con su evidencia."), N("  Verifíquela siempre: es lo que hace el diagnóstico defendible ante el equipo formulador.")], 0, RED),
    ([B("La columna «Revisión humana recomendada» es su cola de trabajo."), N("  Empiece por ahí.")], 0, RED),
], size=22, gap=20)
band(s, [B("El valor de la herramienta depende de la calidad de la revisión humana posterior."), N("")], bg=BLUELT)

# ═══ 27 · CIERRE VISUAL ════════════════════════════════════════════════
s = blank_slide()
rect(s, 0, 0, EMU_W, EMU_H, BLUE)
rect(s, Inches(1.2), Inches(3.05), Inches(3.4), Inches(0.09), RED)
tb, tf = textbox(s, Inches(1.2), Inches(3.3), Inches(11.0), Inches(1.6))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "Preguntas"
r.font.size = Pt(54); r.font.bold = True; r.font.color.rgb = WHITE
p2 = tf.add_paragraph(); p2.space_before = Pt(14)
set_runs(p2, [W("Agente GPT · Appraisal Checklist de PRODOCs   ·   Ahmed Eid")], 22)



ilo_deck.finalize("Capacitacion_Appraisal_GPT_Ahmed.pptx")
