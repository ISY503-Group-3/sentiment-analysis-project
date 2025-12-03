#!/usr/bin/env python3
"""Write minimal 1x1 PNG placeholders (base64-decoded) into artifacts/ using only stdlib."""
import base64
from pathlib import Path

png_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='
img_bytes = base64.b64decode(png_b64)

out = Path(__file__).parent.parent / 'artifacts'
out.mkdir(parents=True, exist_ok=True)

files = {
    'evaluation_confusion_matrix.png': img_bytes,
    'evaluation_roc_curve.png': img_bytes,
    'evaluation_confidence_hist.png': img_bytes,
}

for name, data in files.items():
    p = out / name
    with open(p, 'wb') as f:
        f.write(data)
print('Wrote minimal PNG placeholders to', out)
