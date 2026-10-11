"""Draw frame.png: the world frame seen from above."""
from PIL import Image, ImageDraw, ImageFont

W, H, SCALE = 640, 520, 300.0  # pixels per meter
image = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(image)
font = ImageFont.load_default(size=15)
small = ImageFont.load_default(size=13)


def px(x, y):
    return W / 2 + x * SCALE, H - 90 - (y + 0.45) * SCALE


draw.rectangle([*px(-0.75, 0.60), *px(0.75, -0.45)], outline=(140, 90, 40), width=3)
draw.text((px(-0.73, 0.58)[0], px(0, 0.58)[1] + 4), "table, top at z = 0.74", fill=(140, 90, 40), font=small)
for name, x in (("left arm base", -0.30), ("right arm base", 0.30)):
    cx, cy = px(x, -0.45)
    draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(60, 60, 60))
    draw.text((cx - 50, cy + 14), f"{name}\n({x:+.2f}, -0.45)", fill=(60, 60, 60), font=small)
cx, cy = px(0, -0.41)
draw.rectangle([cx - 12, cy - 7, cx + 12, cy + 7], outline=(0, 90, 200), width=2)
draw.text((cx - 60, cy - 28), "head camera, z = 1.31", fill=(0, 90, 200), font=small)
ox, oy = px(0, 0)
for label, (dx, dy), color in (("+x (right)", (0.30, 0), (200, 0, 0)), ("+y (forward)", (0, 0.30), (0, 150, 0))):
    ex, ey = px(dx, dy)
    draw.line([ox, oy, ex, ey], fill=color, width=3)
    draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=color)
    draw.text((ex + 8, ey - 8), label, fill=color, font=font)
draw.ellipse([ox - 5, oy - 5, ox + 5, oy + 5], fill="black")
draw.text((ox - 78, oy + 6), "(0, 0)", fill="black", font=small)
draw.text((12, 10), "World frame seen from above. +z points up, out of the picture. Units: meters.", fill="black", font=small)
draw.text((12, H - 24), "The robot stands at the bottom edge and looks toward +y.", fill="black", font=small)
image.save("frame.png")
