"""Deck for the 30-min ILO training block: GPT PRODOC Quality Appraisal (Ahmed).

Assumes the single-sheet Excel ("Resultado Diagnóstico"), i.e. the four agreed
changes are already applied. Palette and layout helpers mirror the house deck in
presentation/gpts_oit_guia/build_pptx.py.
"""

import math
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BLUE   = RGBColor(0x00, 0x3E, 0x7E)
BLUEMD = RGBColor(0x00, 0x72, 0xBC)
BLUELT = RGBColor(0xE0, 0xEC, 0xF6)
RED    = RGBColor(0xD6, 0x00, 0x1C)
GRAY   = RGBColor(0x4A, 0x4A, 0x4A)
GRAYLT = RGBColor(0xF0, 0xF4, 0xF8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

EMU_W, EMU_H = Inches(13.333), Inches(7.5)
EMU_IN = Inches(1)

prs = Presentation()
prs.slide_width, prs.slide_height = EMU_W, EMU_H
BLANK = prs.slide_layouts[6]

# content area
CL, CT = Inches(0.55), Inches(1.28)
CW, CH = Inches(12.23), Inches(5.85)


def _fill(shape, color):
    shape.fill.solid(); shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def rect(slide, l, t, w, h, color):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    _fill(sp, color); sp.shadow.inherit = False
    return sp


def textbox(slide, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    return tb, tf


def set_runs(para, segments, size=20):
    for seg in segments:
        text, color, bold, italic = (list(seg) + [None, False, False])[:4]
        r = para.add_run(); r.text = text
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic
        r.font.color.rgb = color if color else GRAY


def slide_new(title):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, EMU_W, Inches(0.92), BLUE)
    rect(s, 0, Inches(0.92), EMU_W, Inches(0.055), RED)
    tb, tf = textbox(s, Inches(0.45), 0, Inches(12.4), Inches(0.92), MSO_ANCHOR.MIDDLE)
    r = tf.paragraphs[0].add_run(); r.text = title
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = WHITE
    return s


def bullets(slide, items, l=CL, t=CT, w=CW, h=CH, size=20, gap=11):
    tb, tf = textbox(slide, l, t, w, h)
    first = True
    for segs, level, bcol in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level; p.space_after = Pt(gap)
        if bcol is not None:
            b = p.add_run(); b.text = "▪  "
            b.font.size = Pt(size); b.font.color.rgb = bcol
        set_runs(p, segs, size)
    return tb


def table(slide, headers, rows, l, t, w, col_ratios, fsize=16, header_fs=16, fill_to=None):
    n_rows = len(rows) + 1
    total_h = Inches(0.4) if fill_to is None else max(int(fill_to - t), Inches(0.4))
    gt = slide.shapes.add_table(n_rows, len(headers), l, t, w, total_h).table
    if fill_to is not None:
        hdr_h = int(total_h / n_rows * 0.72)
        body_h = int((total_h - hdr_h) / (n_rows - 1))
        gt.rows[0].height = hdr_h
        for i in range(1, n_rows):
            gt.rows[i].height = body_h
    total = sum(col_ratios)
    for i, r in enumerate(col_ratios):
        gt.columns[i].width = Emu(int(w * r / total))
    for j, htext in enumerate(headers):
        c = gt.cell(0, j); c.fill.solid(); c.fill.fore_color.rgb = BLUE
        c.margin_top = Pt(3); c.margin_bottom = Pt(3)
        run = c.text_frame.paragraphs[0].add_run(); run.text = htext
        run.font.bold = True; run.font.color.rgb = WHITE; run.font.size = Pt(header_fs)
    for i, row in enumerate(rows, start=1):
        shade = GRAYLT if i % 2 == 1 else WHITE
        for j, val in enumerate(row):
            c = gt.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = shade
            c.margin_top = Pt(2); c.margin_bottom = Pt(2)
            p = c.text_frame.paragraphs[0]
            if isinstance(val, tuple):
                run = p.add_run(); run.text = val[0]
                run.font.size = Pt(fsize)
                run.font.color.rgb = val[1] if len(val) > 1 else GRAY
                run.font.bold = val[2] if len(val) > 2 else False
            else:
                run = p.add_run(); run.text = str(val)
                run.font.size = Pt(fsize); run.font.color.rgb = GRAY
    return gt


def band(slide, segments, l=CL, t=Inches(6.35), w=CW, size=17, bg=BLUELT):
    box = rect(slide, l, t, w, Inches(0.72), bg)
    box.text_frame.word_wrap = True
    p = box.text_frame.paragraphs[0]
    set_runs(p, segments, size)
    box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    box.text_frame.margin_left = Pt(12); box.text_frame.margin_right = Pt(12)
    return box


def B(t): return (t, BLUE, True)
def R(t): return (t, RED, True)
def N(t): return (t, GRAY, False)
def W(t): return (t, WHITE, False)
def WB(t): return (t, WHITE, True)
def I(t): return (t, GRAY, False, True)


STEP_GROUPS = []


def steps(slide, items, t=CT, size=19, bottom=Inches(6.22)):
    """Numbered row stack: list of (n, titulo, detalle). Fills t..bottom."""
    n_items = len(items)
    gap = Inches(0.10)
    h = int((bottom - t - gap * (n_items - 1)) / n_items)
    top = t
    group = []
    for n, tit, det in items:
        rect(slide, CL, top, Inches(0.86), h, BLUE)
        tb, tf = textbox(slide, CL, top, Inches(0.86), h, MSO_ANCHOR.MIDDLE)
        group.append(None)  # number box: fixed size, excluded from group fit
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(n)
        r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = WHITE
        rect(slide, CL + Inches(0.86), top, CW - Inches(0.86), h, GRAYLT)
        tb2, tf2 = textbox(slide, CL + Inches(1.08), top, CW - Inches(1.35), h, MSO_ANCHOR.MIDDLE)
        p2 = tf2.paragraphs[0]
        set_runs(p2, [B(tit + "  "), N(det)], size)
        group.append(tb2)
        top = top + h + gap
    STEP_GROUPS.append(group)


# ═══ 1 · PORTADA ═══════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
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
bullets(s, [
    ([B("El criterio no se evalúa como una impresión general.")], 0, None),
    ([N("Se descompone en preguntas cerradas que se responden una por una, con evidencia citada:")], 0, None),
], t=CT, h=Inches(1.2), size=21, gap=8)
table(s, ["Test", "Pregunta cerrada", "Respuesta"],
      [[("T1", BLUE, True), "¿Distingue el tipo de enfoque (sensible / responsivo / transformador)?", ("sí / no", GRAY, True)],
       [("T2", BLUE, True), "¿Articula cómo el proyecto cuestiona normas o relaciones de poder?", ("sí / no", GRAY, True)],
       [("T3", BLUE, True), "¿Hay acciones dedicadas a transformar relaciones, no solo a incluir mujeres?", ("sí / no", GRAY, True)]],
      CL, Inches(2.15), CW, [0.75, 6.2, 1.15], fsize=20, header_fs=19, fill_to=Inches(4.75))
tb, tf = textbox(s, CL, Inches(5.00), CW, Inches(1.15))
p = tf.paragraphs[0]
set_runs(p, [B("DECISIÓN:   "), N("T1 ∧ T2 ∧ T3 → "), (("Yes"), RED, True), N("      ·      T1 ∨ T3 (sin cumplir los tres) → "), ("Partial", RED, True), N("      ·      ¬T1 ∧ ¬T3 → "), ("No", RED, True)], 20)
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

# ═══ 17 · PARTE 2 · PORTADILLA ═════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, EMU_W, EMU_H, BLUE)
rect(s, Inches(1.2), Inches(2.75), Inches(3.4), Inches(0.09), RED)
tb, tf = textbox(s, Inches(1.2), Inches(3.0), Inches(11.0), Inches(2.2))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "Parte 2 · La herramienta en uso"
r.font.size = Pt(48); r.font.bold = True; r.font.color.rgb = WHITE
p2 = tf.add_paragraph(); p2.space_before = Pt(16)
set_runs(p2, [W("Cargar  →  seleccionar  →  ejecutar  →  descargar  →  interpretar")], 26)

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
    ("5", "Construye el Excel", "y lo entrega en la conversación"),
], size=20)
band(s, [B("Si es la primera evaluación del día puede tardar más en arrancar:  "), N("el servicio se suspende por inactividad y necesita unos segundos para despertar. Es esperado.")])

# ═══ 21 · EL EXCEL ═════════════════════════════════════════════════════
s = slide_new("Paso 4 · El Excel: hoja «Resultado Diagnóstico»")
bullets(s, [
    ([B("Una sola hoja"), N(", con una fila por criterio evaluado. Es el registro auditable: consérvelo.")], 0, None),
], t=CT, h=Inches(0.75), size=21, gap=5)
table(s, ["Grupo de columnas", "Qué contiene", "Para qué sirve"],
      [[("Identificación", BLUE, True), "ID · Subsección · Criterio · Transversales", "Ubicar el criterio en el Checklist"],
       [("Resultado", BLUE, True), "Respuesta (Yes / Partial / No / Not Found / N/A)", "El diagnóstico del criterio"],
       [("Confianza", BLUE, True), "Estabilidad (%) · Estable (≥80%) · Resultado Alternativo", "Cuánto coincidieron las 10 corridas"],
       [("Sustento", BLUE, True), "Razonamiento (test por test) · Evidencia citada", "Verificar por qué llegó a ese resultado"],
       [("Prioridad", BLUE, True), ("Revisión humana recomendada", RED, True), "Su cola de trabajo"]],
      CL, Inches(2.05), CW, [2.0, 3.7, 2.6], fsize=18, header_fs=18, fill_to=Inches(6.22))
band(s, [B("«Not Found» no es «No».  "), N("«No» afirma que el criterio no se cumple; «Not Found» afirma que el documento no permite determinarlo. La acción es distinta.")])

# ═══ 22 · ANATOMÍA DE UNA FILA ═════════════════════════════════════════
s = slide_new("Cómo leer una fila, en orden")
steps(s, [
    ("1", "Mire la Respuesta", "Yes / Partial / No / Not Found / N/A"),
    ("2", "Mire la Estabilidad", "¿coincidieron las 10 corridas o hubo desacuerdo?"),
    ("3", "Lea la Evidencia citada", "¿ese pasaje realmente sostiene el resultado?"),
    ("4", "Lea el Razonamiento", "test por test: dónde exactamente falló el criterio"),
    ("5", "Decida", "¿brecha de diseño, brecha documental, o error del Agente?"),
], size=21)
band(s, [R("Nunca acepte una respuesta sin abrir la evidencia. "), N("La evidencia es lo que convierte un resultado automático en un diagnóstico defendible.")])

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
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, EMU_W, EMU_H, BLUE)
rect(s, Inches(1.2), Inches(3.05), Inches(3.4), Inches(0.09), RED)
tb, tf = textbox(s, Inches(1.2), Inches(3.3), Inches(11.0), Inches(1.6))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "Preguntas"
r.font.size = Pt(54); r.font.bold = True; r.font.color.rgb = WHITE
p2 = tf.add_paragraph(); p2.space_before = Pt(14)
set_runs(p2, [W("Agente GPT · Appraisal Checklist de PRODOCs   ·   Ahmed Eid")], 22)


# ════════════════════════════════════════════════════════════════════════
# AUTOFIT — grow text to fill each box, shrink only to avoid overflow
# ════════════════════════════════════════════════════════════════════════
CHAR_W, LINE_H = 0.50, 1.18
MAX_F, MIN_F, STEP = 1.55, 0.62, 0.03
SLIDE_H_IN = 7.5


def para_info(tf):
    out = []
    for p in tf.paragraphs:
        text = "".join(r.text for r in p.runs)
        sizes = [r.font.size.pt for r in p.runs if r.font.size is not None]
        if not sizes:
            continue
        sa = p.space_after.pt if p.space_after is not None else 0
        out.append((text, max(sizes), sa, p.level or 0))
    return out


def est_height_in(paras, box_w_in, factor):
    h = 0.06
    for text, size, sa, level in paras:
        s = size * factor
        usable = box_w_in - 0.2 - level * 0.35
        if usable <= 0.3:
            return 99.0
        cpl = max(1, int(usable / (CHAR_W * s / 72)))
        lines = max(1, math.ceil(len(text) / cpl))
        h += lines * LINE_H * s / 72 + sa / 72
    return h


def fit_factor(paras, box_w_in, box_h_in):
    f = MAX_F
    while f > MIN_F and est_height_in(paras, box_w_in, f) > box_h_in:
        f = round(f - STEP, 2)
    return max(f, MIN_F)


def apply_factor(tf, f):
    for p in tf.paragraphs:
        for r in p.runs:
            if r.font.size is not None:
                r.font.size = Pt(max(1, round(r.font.size.pt * f)))
        if p.space_after is not None:
            p.space_after = Pt(p.space_after.pt * f)


def table_est_height_in(tbl, factor):
    h = 0.0
    col_w = [c.width / EMU_IN for c in tbl.columns]
    for row in tbl.rows:
        row_h = 0.32
        for j, cell in enumerate(row.cells):
            paras = para_info(cell.text_frame)
            if paras:
                row_h = max(row_h, est_height_in(paras, col_w[j], factor) + 0.06)
        h += row_h
    return h


def max_bottom_for(slide, shape):
    top, left, right = shape.top, shape.left, shape.left + shape.width
    limit = SLIDE_H_IN - 0.18
    for other in slide.shapes:
        if other is shape or other.top <= top:
            continue
        o_l, o_r = other.left, other.left + other.width
        if o_r > left and o_l < right:
            limit = min(limit, other.top / EMU_IN - 0.10)
    return limit


# step rows: one shared factor per group so every row reads the same size
STEP_BOXES = set()
for group in STEP_GROUPS:
    boxes = [b for b in group if b is not None]
    if not boxes:
        continue
    f = MAX_F
    for b in boxes:
        paras = para_info(b.text_frame)
        if paras:
            f = min(f, fit_factor(paras, b.width / EMU_IN, b.height / EMU_IN))
    for b in boxes:
        apply_factor(b.text_frame, f)
        STEP_BOXES.add(b._element)

for slide in prs.slides:
    for shape in slide.shapes:
        if shape._element in STEP_BOXES:
            continue
        if shape.has_table:
            tbl = shape.table
            avail = max_bottom_for(slide, shape) - shape.top / EMU_IN
            f = MAX_F
            while f > MIN_F and table_est_height_in(tbl, f) > avail:
                f = round(f - STEP, 2)
            for row in tbl.rows:
                for cell in row.cells:
                    apply_factor(cell.text_frame, max(f, MIN_F))
        elif shape.has_text_frame:
            paras = para_info(shape.text_frame)
            if not paras:
                continue
            avail_h = max_bottom_for(slide, shape) - shape.top / EMU_IN
            h_in = min(shape.height / EMU_IN, max(avail_h, 0.4))
            f = fit_factor(paras, shape.width / EMU_IN, h_in)
            apply_factor(shape.text_frame, f)

prs.save("Capacitacion_Appraisal_GPT_Ahmed.pptx")
print(f"Guardado — {len(prs.slides._sldIdLst)} diapositivas")
