"""
Build the Final Presentation (.pptx) directly as OOXML.

python-pptx is unavailable offline, so we assemble the package by hand:
  * clone the theme / slide-master / slide-layouts / TAU logo / doc-props from
    the mid-presentation deck (preserves branding and the 16:9 size);
  * generate every slide part ourselves from the specs in content.py;
  * embed figures from output/plots and the two demo WAV clips.

Run: python scripts/presentation/build_presentation.py
Output: output/presentation/final_presentation.pptx
"""

import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

import content as C

ROOT = Path(__file__).resolve().parent.parent.parent
MIDDECK = ROOT / 'progress files and requierments' / 'Final Project - Mid Presentation.pptx'
OUT = ROOT / 'output' / 'presentation' / 'final_presentation.pptx'

EMU_IN = 914400
SLIDE_W = 12192000
SLIDE_H = 6858000

# Layout grid (EMU)
MARGIN_X = 838200
TITLE_Y = 980000
TITLE_H = 760000
CONTENT_Y = 1860000
CONTENT_W = SLIDE_W - 2 * MARGIN_X
CONTENT_H = 4500000

# Palette
TITLE_CLR = '1F3864'
ACCENT_CLR = '2E5496'
BODY_CLR = '262626'
TAG_CLR = '5B9BD5'

# Parts copied verbatim from the mid-deck (branding + chrome)
COPY_PARTS = [
    '_rels/.rels',
    'docProps/app.xml', 'docProps/core.xml', 'docProps/thumbnail.jpeg',
    'ppt/presProps.xml', 'ppt/viewProps.xml', 'ppt/tableStyles.xml',
    'ppt/theme/theme1.xml',
    'ppt/slideMasters/slideMaster1.xml',
    'ppt/slideMasters/_rels/slideMaster1.xml.rels',
    'ppt/media/image1.jpeg',  # TAU logo (referenced by master)
]
for i in range(1, 12):
    COPY_PARTS.append(f'ppt/slideLayouts/slideLayout{i}.xml')
    COPY_PARTS.append(f'ppt/slideLayouts/_rels/slideLayout{i}.xml.rels')

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
      'xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main"')


# ---------- low-level XML builders ----------

def esc(s):
    return escape(str(s))


def img_size(path):
    with Image.open(path) as im:
        return im.size  # (w, h) px


def fit(img_w, img_h, box_w, box_h):
    """Scale (img_w,img_h) into box preserving aspect; return (cx, cy) EMU."""
    scale = min(box_w / img_w, box_h / img_h)
    return int(img_w * scale), int(img_h * scale)


def txbody_paragraphs(paras, default_sz=1800, color=BODY_CLR, align=None):
    """paras: list of (text, level). Returns <a:p> ... blocks."""
    out = []
    for text, level in paras:
        sz = default_sz if level == 0 else int(default_sz * 0.85)
        bullet = ('<a:buFont typeface="Arial"/><a:buChar char="•"/>'
                  if level == 0 else
                  '<a:buFont typeface="Arial"/><a:buChar char="–"/>')
        algn = f' algn="{align}"' if align else ''
        # no bullet when centered (title-style)
        if align == 'ctr':
            bullet = '<a:buNone/>'
        ppr = (f'<a:pPr marL="{342900 if level==0 else 742950}" '
               f'indent="-342900" lvl="{level}"{algn}>{bullet}</a:pPr>')
        run = (f'<a:r><a:rPr lang="en-US" sz="{sz}" dirty="0">'
               f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
               f'<a:t>{esc(text)}</a:t></a:r>')
        out.append(f'<a:p>{ppr}{run}</a:p>')
    return ''.join(out)


def textbox(sp_id, name, x, y, w, h, body_xml, anchor='t'):
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sp_id}" name="{esc(name)}"/>'
        f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
        f'<p:txBody><a:bodyPr wrap="square" anchor="{anchor}"><a:normAutofit/></a:bodyPr>'
        f'<a:lstStyle/>{body_xml}</p:txBody></p:sp>'
    )


def title_shape(sp_id, text, y=TITLE_Y, h=TITLE_H, sz=3200, color=TITLE_CLR, align='l'):
    body = (f'<a:p><a:pPr algn="{align}"/><a:r>'
            f'<a:rPr lang="en-US" sz="{sz}" b="1" dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
            f'<a:t>{esc(text)}</a:t></a:r></a:p>')
    return textbox(sp_id, 'Title', MARGIN_X, y, CONTENT_W, h, body, anchor='ctr')


def section_tag(sp_id, n):
    """Small rounded tag top-right: 'Section N'."""
    w, h = 1850000, 380000
    x = SLIDE_W - MARGIN_X - w
    y = 360000
    body = (f'<a:p><a:pPr algn="ctr"/><a:r>'
            f'<a:rPr lang="en-US" sz="1100" b="1" dirty="0">'
            f'<a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></a:rPr>'
            f'<a:t>{esc(SECTION_NAMES[n])}</a:t></a:r></a:p>')
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{sp_id}" name="SectionTag"/>'
        f'<p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
        f'<a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom>'
        f'<a:solidFill><a:srgbClr val="{TAG_CLR}"/></a:solidFill></p:spPr>'
        f'<p:txBody><a:bodyPr anchor="ctr"/><a:lstStyle/>{body}</p:txBody></p:sp>'
    )


SECTION_NAMES = {
    2: 'Topic', 3: 'Motivation & Goals', 4: 'Methods',
    5: 'Demonstration', 6: 'Results', 7: 'Future Work', 8: 'Documentation',
}


def picture(sp_id, rid, x, y, cx, cy, name='Picture'):
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{sp_id}" name="{esc(name)}"/>'
        f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def caption_box(sp_id, text, x, y, w, sz=1300, color=BODY_CLR, align='ctr', italic=True):
    it = ' i="1"' if italic else ''
    body = (f'<a:p><a:pPr algn="{align}"/><a:r>'
            f'<a:rPr lang="en-US" sz="{sz}"{it} dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
            f'<a:t>{esc(text)}</a:t></a:r></a:p>')
    return textbox(sp_id, 'Caption', x, y, w, 500000, body, anchor='t')


def table_frame(sp_id, rows, x, y, w, header_clr=ACCENT_CLR):
    n_cols = max(len(r) for r in rows)
    col_w = w // n_cols
    grid = ''.join(f'<a:gridCol w="{col_w}"/>' for _ in range(n_cols))
    row_xml = []
    for ri, row in enumerate(rows):
        is_h = ri == 0
        row_h = 460000 if is_h else 400000
        cells = ''
        for ci in range(n_cols):
            text = row[ci] if ci < len(row) else ''
            clr = 'FFFFFF' if is_h else BODY_CLR
            b = ' b="1"' if is_h else ''
            fill = (f'<a:solidFill><a:srgbClr val="{header_clr}"/></a:solidFill>'
                    if is_h else
                    (f'<a:solidFill><a:srgbClr val="{"EEF3FA" if ri%2 else "FFFFFF"}"/></a:solidFill>'))
            cell_body = (f'<a:p><a:r><a:rPr lang="en-US" sz="1400"{b} dirty="0">'
                         f'<a:solidFill><a:srgbClr val="{clr}"/></a:solidFill></a:rPr>'
                         f'<a:t>{esc(text)}</a:t></a:r></a:p>')
            cells += (f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>{cell_body}</a:txBody>'
                      f'<a:tcPr marL="91440" marR="91440" marT="45720" marB="45720" anchor="ctr">{fill}</a:tcPr></a:tc>')
        row_xml.append(f'<a:tr h="{row_h}">{cells}</a:tr>')
    tbl = (f'<a:tbl><a:tblPr firstRow="1" bandRow="1"/>'
           f'<a:tblGrid>{grid}</a:tblGrid>{"".join(row_xml)}</a:tbl>')
    total_h = sum(460000 if ri == 0 else 400000 for ri in range(len(rows)))
    return (
        f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{sp_id}" name="Table"/>'
        f'<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>'
        f'<p:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{total_h}"/></p:xfrm>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
        f'{tbl}</a:graphicData></a:graphic></p:graphicFrame>'
    )


def audio_pic(sp_id, name, icon_rid, media_rid, audio_rid, x, y, size=700000):
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{sp_id}" name="{esc(name)}">'
        f'<a:hlinkClick r:id="" action="ppaction://media"/></p:cNvPr>'
        f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>'
        f'<p:nvPr><a:audioFile r:link="{audio_rid}"/>'
        f'<p:extLst><p:ext uri="{{DAA4B4D4-6D71-4841-9C94-3DE7FCFB9230}}">'
        f'<p14:media r:embed="{media_rid}"/></p:ext></p:extLst></p:nvPr></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{icon_rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{size}" cy="{size}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def audio_timing(shape_ids):
    """Click-to-play timing tree for one or more audio shapes."""
    pars = []
    cid = 5
    for spid in shape_ids:
        pars.append(
            f'<p:par><p:cTn id="{cid}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst>'
            f'<p:childTnLst><p:par><p:cTn id="{cid+1}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
            f'<p:childTnLst><p:par><p:cTn id="{cid+2}" presetID="1" presetClass="mediacall" presetSubtype="0" '
            f'fill="hold" nodeType="clickEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
            f'<p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr>'
            f'<p:cTn id="{cid+3}" dur="indefinite" fill="hold"/><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>'
            f'</p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
            f'</p:childTnLst></p:cTn></p:par>'
        )
        cid += 10
    body = ''.join(pars)
    return (
        f'<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
        f'<p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq">'
        f'<p:childTnLst>{body}</p:childTnLst></p:cTn>'
        f'<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
        f'<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
        f'</p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>'
    )


def slide_xml(shapes, timing=''):
    spTree = (
        '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
        '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
        '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
        + ''.join(shapes) + '</p:spTree>'
    )
    # Child order in <p:sld> must be: cSld, clrMapOvr, (transition), timing
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<p:sld {NS}>'
        f'<p:cSld>{spTree}</p:cSld>'
        '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>'
        f'{timing}'
        '</p:sld>'
    )


def slide_rels(layout_num, image_rels=None, audio_rels=None):
    """image_rels: list of (rid, target). audio_rels: list of (media_rid, audio_rid, target)."""
    rels = [f'<Relationship Id="rIdL" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout{layout_num}.xml"/>']
    for rid, target in (image_rels or []):
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{target}"/>')
    for media_rid, audio_rid, target in (audio_rels or []):
        rels.append(f'<Relationship Id="{media_rid}" Type="http://schemas.microsoft.com/office/2007/relationships/media" Target="../media/{target}"/>')
        rels.append(f'<Relationship Id="{audio_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/audio" Target="../media/{target}"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + ''.join(rels) + '</Relationships>')


# ---------- per-slide builders ----------

class MediaRegistry:
    """Collects media files to write into ppt/media with unique names."""
    def __init__(self):
        self.files = {}   # arcname -> bytes
        self._seen = {}   # source key -> arcname
        self._n = 0

    def add(self, data, ext, key=None):
        if key and key in self._seen:
            return self._seen[key]
        self._n += 1
        arc = f'media/img{self._n}.{ext}'
        self.files[arc] = data
        if key:
            self._seen[key] = arc
        return arc


def resolve_image(spec_path, extracted):
    """Return (absolute Path, ext). Handles @tokens from the mid-deck."""
    if spec_path == '@playground':
        return extracted['playground'], 'png'
    if spec_path == '@beforeafter':
        return extracted['beforeafter'], 'png'
    p = ROOT / spec_path
    return p, p.suffix.lstrip('.').lower()


def build_title(spec, sid):
    shapes = []
    # Big title centered
    body = (f'<a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="4000" b="1" dirty="0">'
            f'<a:solidFill><a:srgbClr val="{TITLE_CLR}"/></a:solidFill></a:rPr>'
            f'<a:t>{esc(spec["project_title"])}</a:t></a:r></a:p>')
    shapes.append(textbox(sid(), 'MainTitle', 1200000, 2150000, SLIDE_W - 2400000, 1500000, body, anchor='ctr'))
    # Subtitle
    sub = (f'<a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="2000" i="1" dirty="0">'
           f'<a:solidFill><a:srgbClr val="{ACCENT_CLR}"/></a:solidFill></a:rPr>'
           f'<a:t>{esc(spec["subtitle"])}</a:t></a:r></a:p>')
    shapes.append(textbox(sid(), 'Subtitle', 1200000, 3650000, SLIDE_W - 2400000, 900000, sub, anchor='ctr'))
    # Details block
    lines = [
        (f'Project No. {spec["project_number"]}', 1600, True),
        (spec['students'], 1500, False),
        (spec['supervisor'], 1500, False),
        (spec['location'], 1300, False),
    ]
    det = ''
    for text, sz, b in lines:
        det += (f'<a:p><a:pPr algn="ctr"/><a:r>'
                f'<a:rPr lang="en-US" sz="{sz}" b="{1 if b else 0}" dirty="0">'
                f'<a:solidFill><a:srgbClr val="{BODY_CLR}"/></a:solidFill></a:rPr>'
                f'<a:t>{esc(text)}</a:t></a:r></a:p>')
    shapes.append(textbox(sid(), 'Details', 1200000, 4750000, SLIDE_W - 2400000, 1500000, det, anchor='t'))
    return shapes, '', 1, [], []


def build_bullets(spec, sid):
    shapes = [title_shape(sid(), spec['title']), section_tag(sid(), spec['section'])]
    has_side = 'params' in spec
    body_w = int(CONTENT_W * (0.60 if has_side else 1.0))
    align = 'ctr' if spec.get('center') else None
    default_sz = 2000 if spec.get('center') else 1900
    body = txbody_paragraphs(spec['bullets'], default_sz=default_sz, align=align)
    anchor = 'ctr' if spec.get('center') else 't'
    shapes.append(textbox(sid(), 'Body', MARGIN_X, CONTENT_Y, body_w, CONTENT_H, body, anchor=anchor))
    if has_side:
        tx = MARGIN_X + int(CONTENT_W * 0.63)
        tw = int(CONTENT_W * 0.37)
        shapes.append(table_frame(sid(), spec['params'], tx, CONTENT_Y + 200000, tw))
    return shapes, '', 2, [], []


def build_table(spec, sid):
    shapes = [title_shape(sid(), spec['title']), section_tag(sid(), spec['section'])]
    shapes.append(table_frame(sid(), spec['table'], MARGIN_X, CONTENT_Y, CONTENT_W))
    if spec.get('note'):
        shapes.append(caption_box(sid(), spec['note'], MARGIN_X, CONTENT_Y + 2600000,
                                  CONTENT_W, sz=1400, color=ACCENT_CLR))
    return shapes, '', 2, [], []


def build_image(spec, sid, extracted, media):
    shapes = [title_shape(sid(), spec['title']), section_tag(sid(), spec['section'])]
    path, ext = resolve_image(spec['image'], extracted)
    arc = media.add(path.read_bytes(), ext, key=str(path))
    rid = 'rIdImg1'
    box_h = CONTENT_H - (600000 if spec.get('caption') else 0)
    cx, cy = fit(*img_size(path), CONTENT_W, box_h)
    x = MARGIN_X + (CONTENT_W - cx) // 2
    shapes.append(picture(sid(), rid, x, CONTENT_Y, cx, cy))
    if spec.get('caption'):
        shapes.append(caption_box(sid(), spec['caption'], MARGIN_X,
                                  CONTENT_Y + cy + 80000, CONTENT_W, color=ACCENT_CLR))
    return shapes, '', 2, [(rid, arc.split('/')[-1])], []


def build_image_bullets(spec, sid, extracted, media):
    shapes = [title_shape(sid(), spec['title']), section_tag(sid(), spec['section'])]
    body_w = int(CONTENT_W * 0.50)
    body = txbody_paragraphs(spec['bullets'], default_sz=1800)
    shapes.append(textbox(sid(), 'Body', MARGIN_X, CONTENT_Y, body_w, CONTENT_H, body))
    path, ext = resolve_image(spec['image'], extracted)
    arc = media.add(path.read_bytes(), ext, key=str(path))
    rid = 'rIdImg1'
    img_box_x = MARGIN_X + int(CONTENT_W * 0.52)
    img_box_w = int(CONTENT_W * 0.48)
    cx, cy = fit(*img_size(path), img_box_w, CONTENT_H)
    x = img_box_x + (img_box_w - cx) // 2
    y = CONTENT_Y + (CONTENT_H - cy) // 2
    shapes.append(picture(sid(), rid, x, y, cx, cy))
    return shapes, '', 2, [(rid, arc.split('/')[-1])], []


def build_two_image(spec, sid, extracted, media):
    shapes = [title_shape(sid(), spec['title']), section_tag(sid(), spec['section'])]
    rels = []
    gap = 300000
    half = (CONTENT_W - gap) // 2
    box_h = CONTENT_H - 500000
    for idx, (key, capkey, bx) in enumerate([
        ('image_left', 'caption_left', MARGIN_X),
        ('image_right', 'caption_right', MARGIN_X + half + gap),
    ]):
        path, ext = resolve_image(spec[key], extracted)
        arc = media.add(path.read_bytes(), ext, key=str(path))
        rid = f'rIdImg{idx+1}'
        rels.append((rid, arc.split('/')[-1]))
        cx, cy = fit(*img_size(path), half, box_h)
        x = bx + (half - cx) // 2
        shapes.append(picture(sid(), rid, x, CONTENT_Y, cx, cy))
        if spec.get(capkey):
            shapes.append(caption_box(sid(), spec[capkey], bx, CONTENT_Y + box_h + 40000,
                                      half, sz=1300, color=ACCENT_CLR))
    return shapes, '', 2, rels, []


def build_demo(spec, sid, extracted, media):
    shapes = [title_shape(sid(), spec['title']), section_tag(sid(), spec['section'])]
    # Playground screenshot on the left (~62%)
    path, ext = resolve_image(spec['image'], extracted)
    arc = media.add(path.read_bytes(), ext, key=str(path))
    img_rid = 'rIdImg1'
    img_w = int(CONTENT_W * 0.60)
    cx, cy = fit(*img_size(path), img_w, CONTENT_H)
    shapes.append(picture(sid(), img_rid, MARGIN_X, CONTENT_Y + (CONTENT_H - cy) // 2, cx, cy))
    image_rels = [(img_rid, arc.split('/')[-1])]

    # Right column: bullets + two audio icons with labels
    rx = MARGIN_X + int(CONTENT_W * 0.63)
    rw = int(CONTENT_W * 0.37)
    body = txbody_paragraphs(spec['bullets'], default_sz=1500)
    shapes.append(textbox(sid(), 'DemoBody', rx, CONTENT_Y, rw, 1900000, body))

    # Audio icon image (reuse extracted speaker icon)
    icon_path = extracted['audio_icon']
    icon_arc = media.add(icon_path.read_bytes(), 'png', key=str(icon_path))
    icon_rid = 'rIdIcon'
    image_rels.append((icon_rid, icon_arc.split('/')[-1]))

    audio_rels = []
    audio_shape_ids = []
    audio_specs = [
        (spec['audio_before'], spec['label_before'], CONTENT_Y + 2050000),
        (spec['audio_after'], spec['label_after'], CONTENT_Y + 3050000),
    ]
    for i, (apath, label, ay) in enumerate(audio_specs):
        ap = ROOT / apath
        a_arc = media.add(ap.read_bytes(), 'wav', key=str(ap))
        media_rid = f'rIdMedia{i+1}'
        audio_rid = f'rIdAudio{i+1}'
        audio_rels.append((media_rid, audio_rid, a_arc.split('/')[-1]))
        aid = sid()
        audio_shape_ids.append(aid)
        shapes.append(audio_pic(aid, label, icon_rid, media_rid, audio_rid, rx, ay))
        # label next to the icon
        lbl = (f'<a:p><a:r><a:rPr lang="en-US" sz="1500" b="1" dirty="0">'
               f'<a:solidFill><a:srgbClr val="{BODY_CLR}"/></a:solidFill></a:rPr>'
               f'<a:t>{esc(label)}  ▶</a:t></a:r></a:p>')
        shapes.append(textbox(sid(), f'AudioLbl{i}', rx + 800000, ay + 150000, rw - 800000, 500000, lbl, anchor='ctr'))

    timing = audio_timing(audio_shape_ids)
    return shapes, timing, 2, image_rels, audio_rels


BUILDERS = {
    'title': lambda s, sid, ex, md: build_title(s, sid),
    'bullets': lambda s, sid, ex, md: build_bullets(s, sid),
    'table': lambda s, sid, ex, md: build_table(s, sid),
    'image': lambda s, sid, ex, md: build_image(s, sid, ex, md),
    'image_bullets': lambda s, sid, ex, md: build_image_bullets(s, sid, ex, md),
    'two_image': lambda s, sid, ex, md: build_two_image(s, sid, ex, md),
    'demo': lambda s, sid, ex, md: build_demo(s, sid, ex, md),
}


def extract_middeck_assets():
    """Pull the playground screenshot, before/after panel and audio icon from
    the mid-deck media, plus return its copied parts."""
    assets_dir = ROOT / 'output' / 'presentation' / '_assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(MIDDECK) as z:
        mapping = {
            'playground': 'ppt/media/image7.png',
            'beforeafter': 'ppt/media/image6.png',
            'audio_icon': 'ppt/media/image8.png',
        }
        extracted = {}
        for key, src in mapping.items():
            data = z.read(src)
            dst = assets_dir / f'{key}.png'
            dst.write_bytes(data)
            extracted[key] = dst
        copied = {p: z.read(p) for p in COPY_PARTS}
    return extracted, copied


def content_types(n_slides):
    defaults = (
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Default Extension="jpeg" ContentType="image/jpeg"/>'
        '<Default Extension="wav" ContentType="audio/x-wav"/>'
    )
    over = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(1, 12):
        over.append(f'<Override PartName="/ppt/slideLayouts/slideLayout{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>')
    for i in range(1, n_slides + 1):
        over.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            + defaults + ''.join(over) + '</Types>')


def presentation_xml(n_slides):
    sld_ids = ''.join(
        f'<p:sldId id="{256 + i}" r:id="rId{i + 2}"/>' for i in range(n_slides)
    )
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
            f'<p:sldIdLst>{sld_ids}</p:sldIdLst>'
            f'<p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/><p:notesSz cx="6858000" cy="9144000"/>'
            '</p:presentation>')


def presentation_rels(n_slides):
    rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(n_slides):
        rels.append(f'<Relationship Id="rId{i+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>')
    base = n_slides + 2
    rels.append(f'<Relationship Id="rId{base}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>')
    rels.append(f'<Relationship Id="rId{base+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>')
    rels.append(f'<Relationship Id="rId{base+2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    rels.append(f'<Relationship Id="rId{base+3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + ''.join(rels) + '</Relationships>')


def main():
    print('Extracting branding assets from the mid-deck...')
    extracted, copied = extract_middeck_assets()

    media = MediaRegistry()
    slide_parts = {}   # 'ppt/slides/slideN.xml' -> xml
    slide_rel_parts = {}

    print(f'Building {len(C.SLIDES)} slides:')
    for idx, spec in enumerate(C.SLIDES, start=1):
        counter = {'n': 1}

        def sid():
            counter['n'] += 1
            return counter['n']

        kind = spec['kind']
        shapes, timing, layout_num, image_rels, audio_rels = BUILDERS[kind](
            spec, sid, extracted, media)
        slide_parts[f'ppt/slides/slide{idx}.xml'] = slide_xml(shapes, timing)
        slide_rel_parts[f'ppt/slides/_rels/slide{idx}.xml.rels'] = slide_rels(
            layout_num, image_rels, audio_rels)
        title = spec.get('title', spec.get('project_title', kind))
        print(f'  {idx:2d}. [{kind:14s}] {title}')

    n = len(C.SLIDES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        # copied branding parts
        for arc, data in copied.items():
            z.writestr(arc, data)
        # generated package parts
        z.writestr('[Content_Types].xml', content_types(n))
        z.writestr('ppt/presentation.xml', presentation_xml(n))
        z.writestr('ppt/_rels/presentation.xml.rels', presentation_rels(n))
        # slides + rels
        for arc, xml in slide_parts.items():
            z.writestr(arc, xml)
        for arc, xml in slide_rel_parts.items():
            z.writestr(arc, xml)
        # media
        for arc, data in media.files.items():
            z.writestr(f'ppt/{arc}', data)

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f'\nSaved: {OUT} ({size_mb:.2f} MB, {n} slides)')
    if size_mb > 20:
        print('  WARNING: over 20 MB.')


if __name__ == '__main__':
    main()
