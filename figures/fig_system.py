"""Schematic of forward and return paths with antenna selection.

Coordinates are inches; both panels use exactly the same world-x mapping.
No external style or skill files are imported. Run from any directory.
"""
from pathlib import Path
import math
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.transforms import ScaledTranslation
from PIL import Image, ImageOps
OUT = Path(__file__).resolve().parent / 'generated'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 7, 'pdf.fonttype': 42, 'axes.linewidth': 0.6, 'lines.markersize': 3.5})
INK = '#30363b'
C = dict(blue='#496f8a', red='#b16b78', green='#598b78', gray='#555555')
RED_DASH = (0, (5, 2.8))
GREEN_DASH = (0, (3.2, 2.3))
(W, H) = (7, 2.9)
LAMBDA_IN = 0.66
PITCH = 0.5 * LAMBDA_IN
SHIFT = 0.47 * LAMBDA_IN
RESIDUAL = -0.03 * LAMBDA_IN
PF = 1.9
ANT_Y = 1.37
ZERO_Y = 1.67
AMP = 0.175
LEFTS = (0.08, 3.57)
fig = plt.figure(figsize=(W, H), facecolor='white')
TEXTS = []
GEOMETRY = []
ELEMENTS = []
COMPASSES = []

def txt(ax, x, y, s, size=7, **kwargs):
    t = ax.text(x, y, s, fontsize=size, ha=kwargs.pop('ha', 'center'), va=kwargs.pop('va', 'center'), color=kwargs.pop('color', INK), zorder=kwargs.pop('zorder', 20), **kwargs)
    TEXTS.append(t)
    return t

def arr(ax, p, q, color=INK, lw=0.65, scale=5, z=8, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=scale, color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=z, **kw))

def wire(ax, points):
    ax.plot(*zip(*points), color=C['gray'], lw=0.55, zorder=7)
    arr(ax, points[-2], points[-1], C['gray'], 0.55, 4.5, z=8)

def segment_arrow(ax, p, q, t, color, reverse=False):
    (dx, dy) = (q[0] - p[0], q[1] - p[1])
    norm = math.hypot(dx, dy)
    c = (p[0] + t * dx, p[1] + t * dy)
    u = (dx / norm * 0.043, dy / norm * 0.043)
    a = (c[0] - u[0], c[1] - u[1])
    b = (c[0] + u[0], c[1] + u[1])
    arr(ax, b if reverse else a, a if reverse else b, color, 1, 5.6, z=6)

def layer(ax, x, y, w, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.136, boxstyle='round,pad=0.01,rounding_size=0.028', facecolor=color, edgecolor=color, alpha=0.48, lw=0.6, zorder=6))
    txt(ax, x + w / 2, y + 0.068, label)

def station(ax):
    ax.add_patch(FancyBboxPatch((0.045, 0.27), 0.64, 0.78, boxstyle='round,pad=0.008,rounding_size=0.015', facecolor='#f8f9fa', edgecolor=INK, lw=0.85, zorder=4))
    for x in (0.205, 0.365, 0.525):
        ax.plot([x, x], [1.05, ANT_Y], color=INK, lw=0.8, zorder=8)
        ax.add_patch(FancyBboxPatch((x - 0.023, 1.265), 0.046, 0.105, boxstyle='round,pad=0,rounding_size=0.008', facecolor='white', edgecolor=INK, lw=0.65, zorder=9, path_effects=[pe.Stroke(linewidth=2.6, foreground='white'), pe.Normal()]))
    for y in (0.315, 0.345):
        ax.plot([0.1, 0.23], [y, y], color='#555555', lw=0.55, zorder=7)
    layer(ax, 0.115, 0.49, 0.5, 'layer 1', '#b4d0e1')
    layer(ax, 0.115, 0.75, 0.5, 'layer 2', '#b4d0e1')
    txt(ax, 0.365, 0.38, '$x$', 8)
    wire(ax, [(0.365, 0.45), (0.365, 0.475)])
    wire(ax, [(0.365, 0.64), (0.365, 0.735)])
    wire(ax, [(0.365, 0.9), (0.365, 1.25)])
    txt(ax, 0.47, 0.98, '$s$', 8, color=C['blue'])
    txt(ax, 0.365, 0.12, 'Base station')

def phone(ax, selected):
    for ghost in (True, False):
        shift = SHIFT if ghost else 0
        edge = '#555555' if ghost else INK
        dash = (0, (3, 2.4)) if ghost else '-'
        z = 3 if ghost else 4
        ax.add_patch(FancyBboxPatch((1.21 + shift, 0.11), 0.72, 0.99, boxstyle='round,pad=0.012,rounding_size=0.065', facecolor='none' if ghost else '#f8f9fa', edgecolor=edge, lw=0.8, linestyle=dash, zorder=z))
        ax.plot([1.48 + shift, 1.66 + shift], [1.055, 1.055], color=edge, lw=1, ls=dash, zorder=z + 0.1)
        ax.plot([1.49 + shift, 1.65 + shift], [0.155, 0.155], color=edge, lw=1, ls=dash, zorder=z + 0.1)
        row = [PF - 2 * PITCH, PF - PITCH, PF] if selected else [PF]
        for x in row:
            xx = x + shift
            ax.plot([xx, xx], [1.115, ANT_Y], color=edge, lw=0.65, ls=(0, (1.5, 1.8)) if ghost else '-', zorder=7, solid_capstyle='butt', dash_capstyle='butt', path_effects=[pe.Stroke(linewidth=2.4, foreground='white', capstyle='butt'), pe.Normal()])
            ax.plot(xx, ANT_Y, 'o', ms=2.6, mfc='white', mec=edge, mew=0.65, zorder=8)
    layer(ax, 1.26, 0.83, 0.46, 'layer 3', '#e5c0ca')
    layer(ax, 1.26, 0.58, 0.46, 'layer 4', '#cfc4e2')
    txt(ax, 1.49, 0.38, 'loss $L$')
    wire(ax, [(PF, 1.34), (PF, 1.155), (1.37, 1.155), (1.37, 1.005), (1.49, 1.005), (1.49, 0.982)])
    wire(ax, [(1.49, 0.815), (1.49, 0.73)])
    wire(ax, [(1.49, 0.565), (1.49, 0.445)])
    wire(ax, [(1.73, 0.38), (1.83, 0.38), (1.83, 0.985)])
    ax.plot([1.83, 1.83, PF, PF], [0.985, 1.19, 1.19, 1.345], color=C['gray'], lw=0.55, zorder=7)
    txt(ax, 1.8, 0.68, '$\\delta$', 8)
    txt(ax, 1.1, 0.32, '$t_f$', 7.5, color='#555555')
    txt(ax, 1.57, 0.0, 'Terminal')
    txt(ax, 2.19, 0.015, '$t_f+\\tau$', 7.5, color='#555555')
    arr(ax, (0.82, 0.155), (0.82 + SHIFT, 0.155), INK, 0.65, 5)
    txt(ax, 0.82 + SHIFT / 2, 0.045, '$v\\!\\cdot\\!\\tau$', 7.5)

def residual_detail(ax):
    existing_artists = set(ax.get_children())
    (cx, cy, scale) = (2.83, 0.25, 3 * LAMBDA_IN)
    j = cx + scale * -0.03
    (lo, hi) = (cx - scale * 0.25, cx + scale * 0.25)
    ax.fill_between([lo, hi], cy - 0.045, cy + 0.045, color='#a9cbb7', alpha=0.65, lw=0, zorder=2)
    ax.plot([lo, hi], [cy, cy], color=C['green'], lw=0.65, zorder=3)
    for x in (j, cx):
        ax.plot([x, x], [cy - 0.06, cy + 0.1], color=INK, lw=0.8, zorder=5)
    arr(ax, (j - 0.16, cy + 0.075), (j, cy + 0.075), INK, 0.6, 4, z=6)
    arr(ax, (cx + 0.16, cy + 0.075), (cx, cy + 0.075), INK, 0.6, 4, z=6)
    ax.plot([j, cx], [cy + 0.075, cy + 0.075], color=INK, lw=0.6, zorder=6)
    txt(ax, 3.13, cy + 0.085, '$r$', 8)
    txt(ax, 2.56, 0.13, '$j^*$', 7.5, color=C['green'])
    txt(ax, 3.04, 0.13, '$p_f$', 7.5)
    ax.plot([2.64, j], [0.16, cy - 0.055], color=C['green'], lw=0.5)
    ax.plot([2.94, cx], [0.16, cy - 0.055], color=INK, lw=0.5)
    txt(ax, 2.83, 0.44, 'residual detail (enlarged)', 7)
    txt(ax, 2.83, 0.015, 'tolerance $|r|<\\lambda/4$', 7, color=C['green'])
    from matplotlib.transforms import Affine2D
    transform = Affine2D().translate(-2.83, -0.25).scale(0.85).translate(2.83, 0.235)
    for artist in set(ax.get_children()) - existing_artists:
        if isinstance(artist, matplotlib.text.Text):
            (old_x, old_y) = artist.get_position()
            artist.set_position((transform.transform((old_x, old_y))[0], old_y))
        else:
            artist.set_transform(transform + ax.transData)

def lobe(ax, selected):
    u = np.linspace(-0.8, 0.8, 801)
    x = PF + LAMBDA_IN * u
    y = ZERO_Y + AMP * np.cos(2 * np.pi * u)
    ax.fill_between(x, ZERO_Y, y, where=np.abs(u) <= 0.25, interpolate=True, color='#a9cbb7', alpha=0.65, lw=0, zorder=3)
    ax.fill_between(x, ZERO_Y, y, where=y <= ZERO_Y, interpolate=True, color='#e5bec7', alpha=0.46, lw=0, zorder=3)
    ax.plot(x, y, color='#95a89e', alpha=0.85, lw=0.7, zorder=4)
    ax.plot([x[0], x[-1]], [ZERO_Y, ZERO_Y], color='#a8abad', lw=0.5, zorder=4)
    for (u, label) in ((-0.25, '$-\\lambda/4$'), (0.25, '$+\\lambda/4$')):
        xx = PF + u * LAMBDA_IN
        ax.plot([xx, xx], [ZERO_Y - 0.025, ZERO_Y + 0.025], color='#8b9095', lw=0.5, zorder=5)
        label_x = 1.56 if u < 0 else 2.18
        txt(ax, label_x, 1.985, label)
        ax.plot([xx, xx, label_x], [ZERO_Y + 0.03, 1.89, 1.925], color='#8b9095', lw=0.4, zorder=5)
    txt(ax, 2.52, 2.29, 'lower bound $\\cos(kr)$ on Re $\\varphi$', 7)
    txt(ax, 2.36, 2.135, 'mean descent guaranteed: $|r|<\\lambda/4$', 7, color=C['green'])
    ax.plot([PF, PF], [ANT_Y, 1.87], color='#969c9e', lw=0.5, ls=(0, (3, 3)), zorder=5)
    txt(ax, PF - 0.075, 1.94, '$p_f$', 7.5)
    if selected:
        j = PF + RESIDUAL
        ax.plot([j, j], [ANT_Y, 1.875], color=C['green'], lw=0.45, ls=(0, (1.2, 2)), zorder=5)
        for (tail, tip) in ((j - 0.085, j), (PF + 0.085, PF)):
            arr(ax, (tail, 1.88), (tip, 1.88), INK, 0.45, 2.5, z=10)
        ax.plot([j, PF], [1.88, 1.88], color=INK, lw=0.4, zorder=10)
        txt(ax, PF + 0.085, 1.94, '$r$', 7.5)
    else:
        moved = PF + SHIFT
        ax.plot([moved, moved], [ANT_Y, ZERO_Y + AMP * math.cos(2 * math.pi * 0.47)], color=C['red'], lw=0.55, ls=(0, (1.5, 1.8)), zorder=5)

def compass(ax, selected):
    col = C['green'] if selected else C['red']
    o = (2.82, 1.23)
    ax.add_patch(Circle((2.91, 1.34), 0.405, fc='#fafbfa', ec='#d7dbd9', lw=0.6, zorder=1))
    arr(ax, o, (o[0], 1.7), INK, 1.45, 8, z=8)
    txt(ax, 2.68, 1.81, '$\\mathbf{g}$ (true)')
    if selected:
        angle = 12
        end = (o[0] + 0.47 * math.sin(math.radians(angle)), o[1] + 0.47 * math.cos(math.radians(angle)))
        txt(ax, 2.83, 1.94, '$\\mathrm{E}[\\hat{\\mathbf{g}}]$ (mean returned)', color=col)
        ax.add_patch(Arc(o, 0.4, 0.4, theta1=78, theta2=90, color=INK, lw=0.65, zorder=9))
        txt(ax, 3.12, 1.29, '$<90^{\\circ}$', 7.5)
    else:
        end = (3.16, 0.98)
        angle = 90 - math.degrees(math.atan2(end[1] - o[1], end[0] - o[0]))
        txt(ax, 2.83, 0.865, '$\\mathrm{E}[\\hat{\\mathbf{g}}]$ (mean returned)', color=col)
        ax.add_patch(Arc(o, 0.37, 0.37, theta1=-36.4, theta2=90, color=INK, lw=0.65, zorder=9))
        txt(ax, 3.12, 1.43, '$>90^{\\circ}$', 7.5)
    arr(ax, o, end, col, 1.6, 8, z=9)
    ax.plot(*o, 'o', ms=2.5, color=INK, zorder=10)
    txt(ax, 2.83, 0.79 if selected else 0.69, 'Update $-\\mathrm{E}[\\hat{\\mathbf{g}}]$:', 7.5, color=col)
    txt(ax, 2.83, 0.66 if selected else 0.52, 'mean descent' if selected else 'mean ascent possible', 7.5, color=col, fontweight='bold')
    if selected:
        residual_detail(ax)
    COMPASSES.append(dict(panel='b' if selected else 'a', angle_degrees=angle))

def panel(left, selected):
    ax = fig.add_axes([left / W, 0.06 / H, 3.38 / W, 2.78 / H])
    ax.set(xlim=(0, 3.38), ylim=(0, 2.78))
    ax.axis('off')
    name = 'b' if selected else 'a'
    col = C['green'] if selected else C['red']
    dash = GREEN_DASH if selected else RED_DASH
    txt(ax, 0.04, 2.68, '(b) Antenna position selection' if selected else '(a) Antenna fixed on the body', 9, ha='left', fontweight='bold')
    for (y, label, color, ls, reverse) in ((2.445, 'activations (forward)', C['blue'], '-', False), (2.255, 'restored backward pass' if selected else 'aged backward pass', col, dash, True)):
        ax.plot([0.1, 0.44], [y, y], color=color, lw=1.45 if not reverse else 1.15, ls=ls, zorder=5)
        arr(ax, (0.34 if not reverse else 0.2, y), (0.44 if not reverse else 0.1, y), color, 0.8, 5)
        txt(ax, 0.52, y, label, 7.5, ha='left', color=color)
    bs = (0.365, ANT_Y)
    pf = (PF, ANT_Y)
    moved = (PF + SHIFT, ANT_Y)
    ret = (PF + RESIDUAL, ANT_Y) if selected else moved
    scatterers = [(0.84, 1.66), (0.84, 1.24)]
    for s in scatterers:
        fp = [bs, s, pf]
        rp = [bs, s, ret]
        ax.plot(*zip(*fp), color=C['blue'], lw=1.8 if selected else 1.55, zorder=2, solid_capstyle='butt')
        if selected:
            return_transform = ax.transData + ScaledTranslation(0, 0.55 / 72, fig.dpi_scale_trans)
            ax.plot(*zip(*rp), color='white', lw=1.2, ls=(0, (3.2 * 0.7 / 1.2, 2.3 * 0.7 / 1.2)), zorder=2.8, transform=return_transform, dash_capstyle='butt')
        ax.plot(*zip(*rp), color=col, lw=0.7 if selected else 1.05, ls=dash, zorder=3, dash_capstyle='butt', transform=return_transform if selected else ax.transData)
        segment_arrow(ax, s, pf, 0.6, C['blue'])
        segment_arrow(ax, bs, s, 0.52, col, True)
        ax.add_patch(Polygon([(s[0] - 0.04, s[1] - 0.03), (s[0] - 0.025, s[1] + 0.04), (s[0] + 0.035, s[1] + 0.03), (s[0] + 0.044, s[1] - 0.027)], fc='#c8cccf', ec='#7b8287', lw=0.55, zorder=8))
        GEOMETRY.append(dict(panel=name, forward=fp, returned=rp))
    txt(ax, 0.81, 1.85, 'static scatterers')
    txt(ax, 0.99, 1.44, 'matched path phases' if selected else 'different path phases', color=col)
    station(ax)
    phone(ax, selected)
    lobe(ax, selected)
    ax.plot(*pf, 'o', ms=4.2, mfc='white', mec=C['blue'], mew=1, zorder=11)
    ax.plot(*ret, 's', ms=3, mfc=col, mec='white', mew=0.3, zorder=13)
    if selected:
        ax.plot(*moved, 'o', ms=3.5, mfc='white', mec=INK, mew=0.7, zorder=12)
        ax.add_patch(FancyArrowPatch(moved, ret, connectionstyle='arc3,rad=.65', arrowstyle='-|>', mutation_scale=5, linewidth=0.7, color=col, shrinkA=3, shrinkB=3, zorder=12))
        txt(ax, 2.36, 1.425, 'switch', color=col)
        txt(ax, 2.08, 1.21, '$j^*$', 8, color=col)
        ax.plot([2.025, ret[0]], [1.29, 1.345], color=col, lw=0.5, zorder=11)
    else:
        txt(ax, 2.43, 1.025, '$p_f+v\\tau$', 7, color=col)
        ax.plot([2.43, 2.39, ret[0] + 0.025], [1.095, 1.285, 1.355], color=col, lw=0.45, zorder=11)
    txt(ax, 0.94, 0.8, '$H_r\\approx H_f$' if selected else '$H_r\\ne H_f$', 8, color=col)
    compass(ax, selected)
    world = dict(p_f=0, shifted_forward=0.47, selected=-0.03 if selected else 0.47, original_row=[-1, -0.5, 0] if selected else [0], translated_row=[-0.53, -0.03, 0.47] if selected else [0.47], zero_ticks=[-0.25, 0.25])
    ELEMENTS.append(dict(panel=name, world_lambda=world, pixels_per_lambda=LAMBDA_IN * 300, pixel_x={k: (left + PF + LAMBDA_IN * v) * 300 for (k, v) in world.items() if isinstance(v, (int, float))}, row_pixel_x=[(left + PF + LAMBDA_IN * v) * 300 for v in world['translated_row']], tick_pixel_x=[(left + PF + LAMBDA_IN * v) * 300 for v in world['zero_ticks']], antenna_pixel_y_from_top=(H - 0.06 - ANT_Y) * 300, original_forward=pf, shifted_original=moved, selected=ret))
panel(LEFTS[0], False)
panel(LEFTS[1], True)
fig.add_artist(Line2D([0.5, 0.5], [0.065, 0.95], transform=fig.transFigure, color='#e0e3e4', lw=0.6))
fig.savefig(OUT / 'fig1.pdf')
fig.savefig(OUT / 'fig1.png', dpi=300)
ImageOps.grayscale(Image.open(OUT / 'fig1.png')).save(OUT / 'fig1_grayscale.png', dpi=(300, 300))
fig.set_dpi(300)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
clipped = []
overlaps = []
for (i, t) in enumerate(TEXTS):
    b = t.get_window_extent(renderer)
    if not fig.bbox.contains(b.x0, b.y0) or not fig.bbox.contains(b.x1, b.y1):
        clipped.append(t.get_text())
    for u in TEXTS[i + 1:]:
        if b.overlaps(u.get_window_extent(renderer)):
            overlaps.append([t.get_text(), u.get_text()])
line_overlaps = []
for t in TEXTS:
    box = t.get_window_extent(renderer)
    for line in t.axes.lines:
        path = line.get_path().transformed(line.get_transform())
        if path.intersects_bbox(box, filled=False):
            line_overlaps.append(t.get_text())
right_texts = [t for t in TEXTS if t.axes == fig.axes[1]]
caption = next((t for t in right_texts if t.get_text() == 'residual detail (enlarged)'))
conclusion = next((t for t in right_texts if t.get_text() == 'mean descent'))
inset_gap_pt = (conclusion.get_window_extent(renderer).y0 - caption.get_window_extent(renderer).y1) * 72 / 300
pdf = (OUT / 'fig1.pdf').read_bytes()
media = [float(v) for v in re.search(b'/MediaBox\\s*\\[([^]]+)\\]', pdf)[1].split()]
images = len(re.findall(b'/Subtype\\s*/Image\\b', pdf))
fonts = sorted(set((v.decode() for v in re.findall(b'/BaseFont\\s*/([^\\s/<>()]+)', pdf))))
report = dict(figure_inches=[W, H], png_pixels=list(Image.open(OUT / 'fig1.png').size), png_dpi=list(Image.open(OUT / 'fig1.png').info['dpi']), min_text_pt=min((t.get_fontsize() for t in TEXTS)), text_outside_canvas=clipped, text_overlaps=overlaps, text_line_intersections=line_overlaps, inset_conclusion_gap_pt=inset_gap_pt, return_display_offset_pt=0.55, labels=[t.get_text() for t in TEXTS], geometry=GEOMETRY, elements=ELEMENTS, compasses=COMPASSES, lobe_height_inches=2 * AMP, lobe_range_lambda=[-0.8, 0.8], wavelength_inches=LAMBDA_IN, pitch_inches=PITCH, translation_inches=SHIFT, residual_inches=RESIDUAL, pdf_media_box=media, pdf_image_objects=images, pdf_base_fonts=fonts, pdf_embedded_truetype=b'/FontFile2' in pdf, layer_height_inches=0.136, layer_height_r4_inches=0.17, detail_scale_relative_to_main=2.55, detail_residual_pixels=abs(RESIDUAL) * 2.55 * 300, phone_body_inches=[0.72, 0.99], cabinet_body_inches=[0.64, 0.78])
(OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
assert not clipped, clipped
assert not overlaps, overlaps
assert not line_overlaps, line_overlaps
assert inset_gap_pt >= 4
assert report['min_text_pt'] >= 7
assert media == [0, 0, 504, 208.8]
assert images == 0 and b'/FontFile2' in pdf
assert all((p['forward'][:-1] == p['returned'][:-1] for p in GEOMETRY))
assert abs(SHIFT - PITCH - RESIDUAL) < 1e-12
assert ELEMENTS[0]['pixels_per_lambda'] == ELEMENTS[1]['pixels_per_lambda'] == 198
assert COMPASSES[0]['angle_degrees'] > 90 and COMPASSES[1]['angle_degrees'] < 90
print(OUT / 'fig1.pdf')
