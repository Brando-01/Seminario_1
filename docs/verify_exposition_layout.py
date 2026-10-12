"""Render the actual Word-exported PDF and print page-level layout evidence."""
from pathlib import Path
import argparse
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='outputs/exposicion_20261012')
folder = root / parser.parse_args().folder
pdf = pdfium.PdfDocument(folder / 'guide.pdf')
thumbs = []
for index in range(len(pdf)):
    page = pdf[index]
    text = page.get_textpage().get_text_range()
    print(f'PAGE {index + 1}: {len(text)} chars | {text[:160]!r}')
    rendered = page.render(scale=1.65).to_pil().convert('RGB')
    rendered.save(folder / f'page-{index + 1:02d}.png')
    thumb = rendered.copy()
    thumb.thumbnail((425, 550))
    tile = Image.new('RGB', (445, 585), 'white')
    tile.paste(thumb, (10, 25))
    ImageDraw.Draw(tile).text((10, 5), f'PAGE {index + 1}', fill='black')
    thumbs.append(tile)
for start in range(0, len(thumbs), 4):
    sheet = Image.new('RGB', (890, 1170), '#dddddd')
    for offset, thumb in enumerate(thumbs[start:start + 4]):
        sheet.paste(thumb, ((offset % 2) * 445, (offset // 2) * 585))
    sheet.save(folder / f'contact-{start // 4 + 1:02d}.png')
print(f'{len(pdf)} pages rendered')
