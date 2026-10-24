"""Render the focused handoff with portrait text and landscape figure pages."""
from pathlib import Path
import re
import html

from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,NextPageTemplate
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'research_handoff'
RUN=BASE/'runs/branch_b_expanded_local_v2'
NAVY=colors.HexColor('#17324d');TEAL=colors.HexColor('#117c83');GRAY=colors.HexColor('#536473')


def markup(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`(.+?)`',r'<font size="9">\1</font>',s)
    return s


def main():
    for name,file in [('Body','Arial.ttf'),('Body-Bold','Arial Bold.ttf'),('Body-Italic','Arial Italic.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'/System/Library/Fonts/Supplemental/'+file))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Italic',boldItalic='Body-Bold')
    styles={
        'body':ParagraphStyle('body',fontName='Body',fontSize=10.4,leading=14.5,spaceAfter=10,textColor=NAVY,wordWrap='CJK'),
        'h1':ParagraphStyle('h1',fontName='Body-Bold',fontSize=18,leading=23,spaceAfter=17,textColor=NAVY),
        'h2':ParagraphStyle('h2',fontName='Body-Bold',fontSize=13,leading=18,spaceAfter=15,textColor=TEAL),
        'cell':ParagraphStyle('cell',fontName='Body',fontSize=7.7,leading=10,textColor=NAVY,wordWrap='CJK'),
        'head':ParagraphStyle('head',fontName='Body-Bold',fontSize=7.7,leading=10,textColor=colors.white,wordWrap='CJK')}
    def decorate(c,doc):
        w,h=doc.pagesize
        c.setStrokeColor(TEAL);c.setLineWidth(2);c.line(40,h-30,w-40,h-30)
        c.setFont('Body-Bold',8);c.setFillColor(GRAY);c.drawString(40,h-23,'BRANCH B  /  EXPANDED LOCAL-SEQUENCE VALIDATION')
        c.setStrokeColor(colors.HexColor('#c9d6dd'));c.setLineWidth(.5);c.line(40,36,w-40,36)
        c.setFont('Body',8);c.drawString(40,23,'Frozen algorithms | Five development scenes | Repository evidence')
        c.drawRightString(w-40,23,str(doc.page))
    def portrait(c,doc):doc.pagesize=A4;c.setPageSize(A4);decorate(c,doc)
    def wide(c,doc):doc.pagesize=landscape(A4);c.setPageSize(landscape(A4));decorate(c,doc)
    templates=[]
    for name,size,fn in [('portrait',A4,portrait),('landscape',landscape(A4),wide)]:
        templates.append(PageTemplate(id=name,frames=[Frame(40,47,size[0]-80,size[1]-98,id=name)],onPage=fn,pagesize=size))
    doc=BaseDocTemplate(str(ROOT/'Branch_B_Research_Handoff.pdf'),pagesize=A4,pageTemplates=templates,
        title='Branch B Research Handoff: Expanded Local Validation',author='Branch B Research Project',pageCompression=1)
    story=[];parts=(ROOT/'Branch_B_Research_Handoff.md').read_text().split('<!-- PAGE BREAK -->')
    page_map=[]
    for idx,part in enumerate(parts):
        isfigure='![' in part;size=landscape(A4) if isfigure else A4;avail=size[0]-92
        if idx:
            story.extend([NextPageTemplate('landscape' if isfigure else 'portrait'),PageBreak()])
        lines=part.strip().splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            if line.startswith('# '):story.append(Paragraph(markup(line[2:]),styles['h1']));i+=1;continue
            if line.startswith('## '):story.append(Paragraph(markup(line[3:]),styles['h2']));i+=1;continue
            if line.startswith('|'):
                raw=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    row=[x.strip() for x in lines[i].strip().strip('|').split('|')];i+=1
                    if not all(re.match(r'^:?-+:?$',x) for x in row):raw.append(row)
                n=len(raw[0]);weights=[1.0]*n
                if n>3:weights[0]=1.25
                if n>3 and raw[0][1] in ['Arm','Policy']:weights[1]=1.8
                if n==2:weights=[.37,.63]
                widths=[avail*w/sum(weights) for w in weights]
                cells=[[Paragraph(markup(x),styles['head' if j==0 else 'cell']) for x in row] for j,row in enumerate(raw)]
                t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f6'),colors.white]),
                    ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                    ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
                story.extend([t,Spacer(1,13)]);continue
            if line.startswith('!['):
                match=re.match(r'!\[(.*?)\]\((.*?)\)',line);path=(ROOT/match[2]).resolve()
                w,h=PILImage.open(path).size;hh=avail*h/w
                if hh>size[1]-300:hh=size[1]-300;ww=hh*w/h
                else:ww=avail
                story.extend([Image(str(path),width=ww,height=hh),Spacer(1,14)]);i+=1;continue
            para=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','![')):
                para.append(lines[i].strip());i+=1
            story.append(Paragraph(markup(' '.join(para)),styles['body']))
    doc.build(story)
    print('Rendered',ROOT/'Branch_B_Research_Handoff.pdf')


if __name__=='__main__':main()
