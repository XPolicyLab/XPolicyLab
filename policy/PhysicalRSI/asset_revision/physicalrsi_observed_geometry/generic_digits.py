"""Digit identity from generic OpenCV/DejaVu fonts, without asset geometry."""
import cv2
import numpy as np
from functools import lru_cache

def normalize(mask):
 y,x=np.nonzero(mask)
 if len(x)==0:raise ValueError('Empty glyph')
 a=mask[y.min():y.max()+1,x.min():x.max()+1].astype(np.uint8);h,w=a.shape;s=min(28/h,28/w);a=cv2.resize(a,(max(1,round(w*s)),max(1,round(h*s))),interpolation=cv2.INTER_AREA)>.4
 out=np.zeros((32,32),bool);h,w=a.shape;out[(32-h)//2:(32-h)//2+h,(32-w)//2:(32-w)//2+w]=a;return out
def holes(mask):
    contours,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:return 0
    return sum(h[3]>=0 and cv2.contourArea(c)>2 for c,h in zip(contours,hierarchy[0]))


@lru_cache(maxsize=1)
def silhouette_bank():
    """Generic 0–9 font silhouettes at the controller's existing yaw samples.

    Glyph pixels have no metric scale and are not rendered from task CAD.
    This bank supplies identity evidence only, never object height or size.
    """
    rows=[]
    for digit in range(10):
        for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
                     cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
                     cv2.FONT_HERSHEY_COMPLEX_SMALL):
            for thickness in (1,2,3):
                glyph=np.zeros((101,101),np.uint8)
                (width,height),baseline=cv2.getTextSize(str(digit),font,2.,thickness)
                cv2.putText(glyph,str(digit),((101-width)//2,(101+height)//2),
                            font,2.,255,thickness,cv2.LINE_AA)
                for angle in range(-50,51,5):
                    rotated=cv2.warpAffine(glyph,cv2.getRotationMatrix2D((50,50),angle,1),(101,101))
                    mask=normalize(rotated>100)
                    mask.setflags(write=False)
                    rows.append((digit,angle,mask))
    # Printed font families cover broad/rounded glyphs absent from thin
    # Hershey strokes. Pixel stroke widths carry no physical object scale.
    from pathlib import Path
    from PIL import Image,ImageDraw,ImageFont
    fonts=Path(__file__).with_name('fonts')
    for name in ('DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSansCondensed-Bold.ttf',
                 'DejaVuSerif-Bold.ttf','DejaVuSansMono-Bold.ttf'):
        font=ImageFont.truetype(str(fonts/name),64)
        for digit in range(10):
            for stroke in (0,2,4,6):
                image=Image.new('L',(128,128));draw=ImageDraw.Draw(image)
                box=draw.textbbox((0,0),str(digit),font=font,stroke_width=stroke)
                draw.text(((128-box[2]-box[0])/2,(128-box[3]-box[1])/2),str(digit),
                          font=font,fill=255,stroke_width=stroke,stroke_fill=255)
                glyph=np.asarray(image)
                for angle in range(-50,51,5):
                    rotated=cv2.warpAffine(glyph,cv2.getRotationMatrix2D((64,64),angle,1),(128,128))
                    mask=normalize(rotated>100);mask.setflags(write=False)
                    rows.append((digit,angle,mask))
    return tuple(rows)

bank={d:[] for d in range(1,10)}
for d in bank:
 for font in [cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL]:
  for thickness in [1,2,3]:
   for scale in [.5,.75,1.,1.5,2.]:
    a=np.zeros((100,100),np.uint8);cv2.putText(a,str(d),(10,80),font,scale,255,thickness,cv2.LINE_AA);bank[d].append(normalize(a>100))


def classify_digit(region):
    image=np.asarray(region)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Current RGB uint8 digit crop required')
    glyph=normalize(np.max(image,axis=2)<100)
    hole_count=holes(glyph)
    scores={digit:min(float(np.mean(glyph!=template))+.1*abs(hole_count-holes(template)) for template in templates)
            for digit,templates in bank.items()}
    return min(scores,key=scores.get)
