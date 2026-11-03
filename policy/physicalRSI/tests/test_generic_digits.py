from pathlib import Path
import sys
import numpy as np
import pytest
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry.generic_digits import classify_digit


@pytest.mark.parametrize('size',[20,32,48])
def test_unseen_font_digits(size):
    path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not path.exists():pytest.skip('Independent test font unavailable')
    font=ImageFont.truetype(str(path),size)
    for digit in range(1,10):
        image=Image.new('RGB',(80,80),'white')
        ImageDraw.Draw(image).text((10,5),str(digit),font=font,fill='black')
        assert classify_digit(np.asarray(image))==digit


def test_blank_card_does_not_invent_digit():
    with pytest.raises(ValueError,match='Empty glyph'):
        classify_digit(np.full((30,30,3),255,np.uint8))
