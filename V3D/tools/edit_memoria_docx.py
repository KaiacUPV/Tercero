from __future__ import annotations

import copy
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

W = f"{{{NS['w']}}}"
M = f"{{{NS['m']}}}"


def q(tag: str) -> str:
    return W + tag


def paragraph_text(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        if node.tag in {q("t"), M + "t"}:
            parts.append(node.text or "")
        elif node.tag == q("tab"):
            parts.append("\t")
        elif node.tag == q("br"):
            parts.append("\n")
    return "".join(parts)


def normalized_text(p: ET.Element) -> str:
    return " ".join(paragraph_text(p).split())


def first_text_node(p: ET.Element) -> ET.Element | None:
    for node in p.iter(q("t")):
        return node
    return None


def set_all_text(p: ET.Element, text: str) -> None:
    first = True
    for node in list(p.iter(q("t"))):
        node.text = text if first else ""
        first = False
    if first:
        r = ET.SubElement(p, q("r"))
        t = ET.SubElement(r, q("t"))
        t.text = text


def set_style(p: ET.Element, style_id: str) -> None:
    p_pr = p.find(q("pPr"))
    if p_pr is None:
        p_pr = ET.Element(q("pPr"))
        p.insert(0, p_pr)
    p_style = p_pr.find(q("pStyle"))
    if p_style is None:
        p_style = ET.Element(q("pStyle"))
        p_pr.insert(0, p_style)
    p_style.set(q("val"), style_id)


def clone_para(template: ET.Element, text: str, style: str | None = None) -> ET.Element:
    p = copy.deepcopy(template)
    set_all_text(p, text)
    if style:
        set_style(p, style)
    return p


def empty_para() -> ET.Element:
    return ET.Element(q("p"))


def page_break_para() -> ET.Element:
    p = ET.Element(q("p"))
    r = ET.SubElement(p, q("r"))
    br = ET.SubElement(r, q("br"))
    br.set(q("type"), "page")
    return p


def replace_body_children(body: ET.Element, start: int, end: int, new_children: list[ET.Element]) -> None:
    children = list(body)
    for child in children[start:end]:
        body.remove(child)
    insert_at = start
    for child in new_children:
        body.insert(insert_at, child)
        insert_at += 1


def find_child_index_by_text(body: ET.Element, pattern: str) -> int | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for i, child in enumerate(list(body)):
        if child.tag == q("p") and regex.search(normalized_text(child)):
            return i
    return None


def insert_after_paragraph(body: ET.Element, idx: int, new_paras: list[ET.Element]) -> None:
    insert_at = idx + 1
    for p in new_paras:
        body.insert(insert_at, p)
        insert_at += 1


def main() -> None:
    src = Path("memoria_SMII_Kai_Aoiz_Canillas_work.docx")
    out = Path("memoria_SMII_Kai_Aoiz_Canillas_editada.docx")
    tmp_dir = Path("tools/_docx_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    with ZipFile(src) as zin:
        zin.extractall(tmp_dir)

    document_path = tmp_dir / "word" / "document.xml"
    tree = ET.parse(document_path)
    root = tree.getroot()
    body = root.find(q("body"))
    if body is None:
        raise RuntimeError("No document body found")

    children = list(body)
    title_template = children[39]
    entry_template = children[40]

    toc_entries = [
        ("1. Resumen de las ideas clave", "6"),
        ("2. Introducción", "6"),
        ("3. Objetivos", "7"),
        ("4. Planificación", "7"),
        ("5. Desarrollo", "9"),
        ("5.1 Elección del método de detección", "9"),
        ("5.2 Arquitectura inicial", "9"),
        ("5.3 Calibración de la cámara", "9"),
        ("5.4 Representación de la recta en tiempo real", "10"),
        ("5.5 Cubo completo", "11"),
        ("5.6 Suavizado", "12"),
        ("5.7 Creación del entorno 3D para el juego", "13"),
        ("5.8 Efectos de sonido y reconocimiento de voz", "14"),
        ("5.9 Modo de realidad virtual", "15"),
        ("6. Conclusiones", "16"),
        ("7. Bibliografía", "17"),
    ]
    figure_entries = [
        ("Figura 1. Arquitectura general del sistema", "9"),
        ("Figura 2. Calibración mediante tablero de ajedrez", "10"),
        ("Figura 3. Primera representación de la recta", "11"),
        ("Figura 4. Diseño final en realidad aumentada", "12"),
        ("Figura 5. Sable en el entorno 3D", "13"),
        ("Figura 6. Modo de cortar cubos", "14"),
        ("Figura 7. Modo combate", "14"),
        ("Figura 8. Modo realidad virtual", "15"),
        ("Figura 9. Diagrama de comunicación para VR", "16"),
    ]
    table_entries = [
        ("Tabla 1. Planificación original", "7"),
        ("Tabla 2. Comparación de la planificación prevista con la real", "8"),
    ]
    formula_entries = [
        ("Fórmula 1. Cálculo del centro de la cara 17", "10"),
        ("Fórmula 2. Centro de la cara en el sistema local", "10"),
        ("Fórmula 3. Recta normal de la cara 17", "11"),
        ("Fórmula 4. Rotación del cubo desde una cara visible", "11"),
        ("Fórmula 5. Traslación del cubo desde una cara visible", "11"),
        ("Fórmula 6. Suavizado exponencial de la traslación", "12"),
    ]

    def index_block(title: str, entries: list[tuple[str, str]]) -> list[ET.Element]:
        block = [clone_para(title_template, title)]
        for name, page in entries:
            block.append(clone_para(entry_template, f"{name}\t{page}", "BodyText"))
        return block

    front_matter = [
        page_break_para(),
        *index_block("Índice general", toc_entries),
        page_break_para(),
        *index_block("Índice de figuras", figure_entries),
        page_break_para(),
        *index_block("Índice de tablas", table_entries),
        page_break_para(),
        *index_block("Índice de fórmulas", formula_entries),
        page_break_para(),
    ]
    replace_body_children(body, 28, 50, front_matter)

    # Text and heading normalization.
    replacements = {
        "Resumen de las ideas clave": ("1. Resumen de las ideas clave", "Heading1"),
        "Introducción": ("2. Introducción", "Heading1"),
        "Objetivos": ("3. Objetivos", "Heading1"),
        "Planificación": ("4. Planificación", "Heading1"),
        "Desarrollo": ("5. Desarrollo", "Heading1"),
        "5.1 Elección del método de detección": ("5.1 Elección del método de detección", "Heading2"),
        "5.2 Arquitectura inicial": ("5.2 Arquitectura inicial", "Heading2"),
        "5.3 CALIBRACIÓN DE LA CÁMARA": ("5.3 Calibración de la cámara", "Heading2"),
        "5.4 Representación de la recta en tiempo real": ("5.4 Representación de la recta en tiempo real", "Heading2"),
        "5.5 SUAVIZADO": ("5.6 Suavizado", "Heading2"),
        "5.6 CREACIÓN DEL ENTORNO 3D PARA EL JUEGO": ("5.7 Creación del entorno 3D para el juego", "Heading2"),
        "5.7 Efectos de sonido y reconocimiento de voz": ("5.8 Efectos de sonido y reconocimiento de voz", "Heading2"),
        "Modo de realidad virtual": ("5.9 Modo de realidad virtual", "Heading2"),
        "Bibliografía": ("7. Bibliografía", "Heading1"),
    }

    for p in body.findall(q("p")):
        txt = normalized_text(p)
        if txt in replacements:
            new_text, style = replacements[txt]
            set_all_text(p, new_text)
            set_style(p, style)
        elif txt.startswith("5.4 CUBO COMPLETO"):
            rest = txt.replace("5.4 CUBO COMPLETO", "", 1).strip()
            rest = rest.replace("Esta método", "Este método")
            rest = rest.replace("lo diferente poses", "las diferentes poses")
            set_all_text(p, "5.5 Cubo completo")
            set_style(p, "Heading2")
            idx = list(body).index(p)
            insert_after_paragraph(body, idx, [clone_para(entry_template, rest, "BodyText")])
        elif txt.startswith("6. CONCLUSIONES"):
            rest = txt.replace("6. CONCLUSIONES", "", 1).strip()
            set_all_text(p, "6. Conclusiones")
            set_style(p, "Heading1")
            idx = list(body).index(p)
            if rest:
                insert_after_paragraph(body, idx, [clone_para(entry_template, rest, "BodyText")])

    # Formula captions.
    formula_captions = {
        "C= (P1 + P2 + P3 + P4) / 4": "Fórmula 1. Cálculo del centro de la cara 17.",
        "C = 0, 0, 0": "Fórmula 2. Centro de la cara en el sistema local.",
        "r(t) = C + t·n": "Fórmula 3. Recta normal de la cara 17.",
        "Rcam,cubo = Rcam,cara · Rcara,cubo^T": "Fórmula 4. Rotación del cubo desde una cara visible.",
        "tcam,cubo = tcam,cara - Rcam,cubo · tcara,cubo": "Fórmula 5. Traslación del cubo desde una cara visible.",
        "t_suavizada = (1 - α) · t_anterior + α · t_medida": "Fórmula 6. Suavizado exponencial de la traslación.",
    }
    inserted = set()
    for p in list(body.findall(q("p"))):
        txt = normalized_text(p)
        if txt in formula_captions and txt not in inserted:
            idx = list(body).index(p)
            insert_after_paragraph(body, idx, [clone_para(entry_template, formula_captions[txt], "BodyText")])
            inserted.add(txt)

    # Caption fixes.
    caption_replacements = {
        "Figura 1. Arquitectura general del sistema.": "Figura 1. Arquitectura general del sistema. Fuente: elaboración propia con Gemini.",
        "Tabla 2. Comparación de la planificación de prevista con la real": "Tabla 2. Comparación de la planificación prevista con la real.",
        "Figura 9. Diagrama de la comunicación para VR.": "Figura 9. Diagrama de comunicación para VR. Fuente: elaboración propia con Gemini.",
    }
    for p in body.findall(q("p")):
        txt = normalized_text(p)
        if txt in caption_replacements:
            set_all_text(p, caption_replacements[txt])
        elif txt == "best_idx = id":
            set_all_text(p, "best_idx = idx")

    # Replace bibliography placeholder.
    bib_idx = find_child_index_by_text(body, r"Bibliografía y referencias")
    if bib_idx is not None:
        refs = [
            "OpenCV. Documentación oficial de OpenCV. https://docs.opencv.org/",
            "OpenCV. Documentación del módulo ArUco. https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html",
            "Vosk. Speech Recognition Toolkit. https://alphacephei.com/vosk/",
            "python-sounddevice. Documentación oficial. https://python-sounddevice.readthedocs.io/",
            "OpenAL Soft. Biblioteca de audio 3D OpenAL. https://openal-soft.org/",
            "Sunshine. Game stream host for Moonlight. https://github.com/LizardByte/Sunshine",
            "Moonlight Game Streaming. Cliente de streaming. https://moonlight-stream.org/",
            "Beat Games. Beat Saber. https://beatsaber.com/",
            "Google. Gemini. Herramienta utilizada para generar las Figuras 1 y 9. https://gemini.google.com/",
        ]
        old = list(body)[bib_idx]
        set_all_text(old, refs[0])
        set_style(old, "BodyText")
        insert_after_paragraph(body, bib_idx, [clone_para(entry_template, ref, "BodyText") for ref in refs[1:]])

    tree.write(document_path, encoding="utf-8", xml_declaration=True)

    if out.exists():
        out.unlink()
    with ZipFile(out, "w", ZIP_DEFLATED) as zout:
        for path in tmp_dir.rglob("*"):
            if path.is_file():
                zout.write(path, path.relative_to(tmp_dir).as_posix())

    shutil.rmtree(tmp_dir)
    print(out)


if __name__ == "__main__":
    main()
