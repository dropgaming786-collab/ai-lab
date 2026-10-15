"""
generate_pdf.py
---------------
Renders docs/QUIZ_PREP_GUIDE.md into a publication-quality PDF using ReportLab Platypus.
Includes professional margins, color themes, running page numbers, callout boxes, and tables.
"""

import os
import sys
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(DOCS_DIR, "QUIZ_PREP_GUIDE.md")
PDF_PATH = os.path.join(DOCS_DIR, "QUIZ_PREP_GUIDE.pdf")

print(f"Reading markdown from: {MD_PATH}")
with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

# Numbered Canvas for "Page X of Y" and Running Header
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "AI-101L Master Quiz Preparation Guide — Labs 01 to 05")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)

        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.drawString(54, 32, "Introduction to Artificial Intelligence (AI-101L) · Student Portfolio")
        self.restoreState()

# Document setup
doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=letter,
    leftMargin=54,
    rightMargin=54,
    topMargin=54,
    bottomMargin=54
)

styles = getSampleStyleSheet()

# Custom styles
primary_color = colors.HexColor("#1a365d")   # Navy
secondary_color = colors.HexColor("#2b6cb0") # Slate Blue
accent_color = colors.HexColor("#2c7a7b")    # Teal
dark_text = colors.HexColor("#2d3748")       # Charcoal
body_font = "Helvetica"
bold_font = "Helvetica-Bold"

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Normal"],
    fontName=bold_font,
    fontSize=22,
    leading=26,
    textColor=primary_color,
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=17,
    textColor=secondary_color,
    spaceAfter=8
)

meta_style = ParagraphStyle(
    "DocMeta",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=9,
    leading=13,
    textColor=colors.HexColor("#4a5568"),
    spaceAfter=12
)

h1_style = ParagraphStyle(
    "Heading1",
    parent=styles["Normal"],
    fontName=bold_font,
    fontSize=14,
    leading=18,
    textColor=primary_color,
    spaceBefore=14,
    spaceAfter=6,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    "Heading2",
    parent=styles["Normal"],
    fontName=bold_font,
    fontSize=11,
    leading=15,
    textColor=secondary_color,
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True
)

h3_style = ParagraphStyle(
    "Heading3",
    parent=styles["Normal"],
    fontName=bold_font,
    fontSize=10,
    leading=13,
    textColor=accent_color,
    spaceBefore=8,
    spaceAfter=3,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName=body_font,
    fontSize=9,
    leading=13,
    textColor=dark_text,
    spaceAfter=5
)

bullet_style = ParagraphStyle(
    "BulletText",
    parent=styles["Normal"],
    fontName=body_font,
    fontSize=9,
    leading=13,
    textColor=dark_text,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=3
)

callout_style = ParagraphStyle(
    "Callout",
    parent=styles["Normal"],
    fontName=body_font,
    fontSize=8.5,
    leading=12.5,
    textColor=colors.HexColor("#1a202c")
)

table_header_style = ParagraphStyle(
    "TableHeader",
    parent=styles["Normal"],
    fontName=bold_font,
    fontSize=8.5,
    leading=11,
    textColor=colors.white,
    alignment=0
)

table_cell_style = ParagraphStyle(
    "TableCell",
    parent=styles["Normal"],
    fontName=body_font,
    fontSize=8,
    leading=11,
    textColor=dark_text
)

def clean_inline_md(text):
    # Bold **text** -> <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic *text* -> <i>text</i>
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Code `code` -> <font name="Courier" color="#b83280">code</font>
    text = re.sub(r'`(.+?)`', r'<font name="Courier" color="#b83280">\1</font>', text)
    # Math and LaTeX cleanups
    text = text.replace(r'\longrightarrow', ' → ')
    text = text.replace(r'\rightarrow', ' → ')
    text = text.replace(r'\dots', '...')
    text = text.replace(r'\quad', '  ')
    text = text.replace(r'\sum', 'Σ')
    text = text.replace(r'\mathcal{D}', 'D')
    text = text.replace(r'\mathcal', '')
    text = text.replace(r'\hat', '')
    text = text.replace(r'\bar', '')
    text = text.replace(r'\max', 'max')
    text = text.replace(r'\min', 'min')
    text = text.replace(r'\ln', 'ln')
    text = text.replace(r'\left', '').replace(r'\right', '')
    text = text.replace(r'\sqrt', '√')
    text = text.replace(r'\propto', '∝')
    text = text.replace(r'\gg', '>>')
    text = text.replace(r'\approx', '≈')
    text = text.replace(r'\le', '≤')
    text = text.replace(r'\ge', '≥')
    text = text.replace(r'\in', '∈')
    text = text.replace(r'\to', '→')
    text = text.replace(r'\times', '×')
    text = text.replace(r'\cdot', '·')
    text = text.replace(r'\mu', 'μ')
    text = text.replace(r'\sigma', 'σ')
    text = text.replace(r'\beta', 'β')
    text = text.replace(r'\Delta', 'Δ')
    text = text.replace(r'\lambda', 'λ')
    text = text.replace(r'\infty', '∞')
    text = text.replace(r'\mathbb{R}', 'R')
    text = text.replace(r'\mathbf', '')
    text = text.replace(r'\text', '')
    text = text.replace(r'\frac', '')
    text = text.replace(r'\{', '{').replace(r'\}', '}')
    text = text.replace('$', '')
    text = text.replace('&', '&amp;')
    # Re-fix tags if & got escaped in tags
    text = text.replace('&amp;lt;', '&lt;').replace('&amp;gt;', '&gt;')
    text = text.replace('&amp;bull;', '•')
    text = text.replace('&bull;', '•')
    return text

lines = md_text.split('\n')
story = []

i = 0
while i < len(lines):
    line = lines[i].strip()
    
    # Empty line
    if not line:
        i += 1
        continue
    
    # Horizontal rule
    if line == '---':
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e0"), spaceAfter=8))
        i += 1
        continue
    
    # H1
    if line.startswith('# '):
        txt = clean_inline_md(line[2:])
        story.append(Paragraph(txt, title_style))
        i += 1
        continue
        
    # H2
    if line.startswith('## '):
        txt = clean_inline_md(line[3:])
        # If this is the main title subtitle
        if "Conceptual Deep Dive" in txt:
            story.append(Paragraph(txt, subtitle_style))
        else:
            story.append(Paragraph(txt, h1_style))
        i += 1
        continue
        
    # H3
    if line.startswith('### '):
        txt = clean_inline_md(line[4:])
        story.append(Paragraph(txt, h2_style))
        i += 1
        continue
        
    # H4
    if line.startswith('#### '):
        txt = clean_inline_md(line[5:])
        story.append(Paragraph(txt, h3_style))
        i += 1
        continue
        
    # Blockquote (Callout box)
    if line.startswith('> '):
        quote_lines = []
        while i < len(lines) and lines[i].strip().startswith('>'):
            quote_lines.append(lines[i].strip()[2:])
            i += 1
        quote_text = clean_inline_md("<br/>".join(quote_lines))
        callout_p = Paragraph(quote_text, callout_style)
        callout_table = Table([[callout_p]], colWidths=[500])
        callout_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ebf8ff")),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LINELEFT', (0,0), (0,0), 3.5, colors.HexColor("#3182ce")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#bee3f8")),
        ]))
        story.append(Spacer(1, 3))
        story.append(callout_table)
        story.append(Spacer(1, 4))
        continue

    # Table
    if line.startswith('|') and '|' in line[1:]:
        table_rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            tline = lines[i].strip()
            # Skip separator line |---|---|
            if set(tline.replace('|', '').strip()) <= {'-', ':', ' '}:
                i += 1
                continue
            cells = [c.strip() for c in tline.split('|')[1:-1]]
            table_rows.append(cells)
            i += 1
            
        if table_rows:
            col_count = len(table_rows[0])
            # Determine column widths
            total_w = 500
            if col_count == 3:
                col_widths = [130, 160, 210]
            elif col_count == 4:
                col_widths = [110, 110, 140, 140]
            elif col_count == 5:
                col_widths = [90, 100, 100, 100, 110]
            elif col_count == 6:
                col_widths = [80, 80, 80, 80, 80, 100]
            else:
                col_widths = [total_w / col_count] * col_count
                
            formatted_table_data = []
            for r_idx, row in enumerate(table_rows):
                fmt_row = []
                for cell in row:
                    clean_cell = clean_inline_md(cell)
                    if r_idx == 0:
                        fmt_row.append(Paragraph(clean_cell, table_header_style))
                    else:
                        fmt_row.append(Paragraph(clean_cell, table_cell_style))
                formatted_table_data.append(fmt_row)
                
            t = Table(formatted_table_data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2b6cb0")),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LEFTPADDING', (0,0), (-1,-1), 5),
                ('RIGHTPADDING', (0,0), (-1,-1), 5),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f7fafc")]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ]))
            story.append(Spacer(1, 4))
            story.append(t)
            story.append(Spacer(1, 6))
        continue

    # Bullet / List item
    if line.startswith('* ') or line.startswith('- ') or re.match(r'^\d+\.\s', line):
        bullet_char = "• "
        if re.match(r'^\d+\.\s', line):
            m = re.match(r'^(\d+\.)\s', line)
            bullet_char = f"{m.group(1)} "
            txt = line[m.end():]
        else:
            txt = line[2:]
        txt = clean_inline_md(bullet_char + txt)
        story.append(Paragraph(txt, bullet_style))
        i += 1
        continue

    # Standard body paragraph
    txt = clean_inline_md(line)
    story.append(Paragraph(txt, body_style))
    i += 1

print(f"Building document with {len(story)} flowables...")
doc.build(story, canvasmaker=NumberedCanvas)
print(f"[SUCCESS] Generated PDF: {PDF_PATH}")
