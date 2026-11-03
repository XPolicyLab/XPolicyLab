from pathlib import Path
import sys
import cv2,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'asset_revision'))
from physicalrsi_observed_geometry.font_identity import FontDigitMatcher


@pytest.fixture(scope='module')
def matcher():return FontDigitMatcher()


def glyph(digit,scale=1):
    mask=np.zeros((120,120),'uint8')
    cv2.putText(mask,str(digit),(20,90),cv2.FONT_HERSHEY_SIMPLEX,2.5,255,3,cv2.LINE_AA)
    return cv2.resize(mask,None,fx=scale,fy=scale)>100


def test_no_asset_load_or_world_position_dependence(monkeypatch,matcher):
    def forbidden(*args,**kwargs):raise AssertionError('Object asset read attempted')
    monkeypatch.setattr(np,'load',forbidden)
    assert not hasattr(matcher,'mesh') and not hasattr(matcher,'render')
    a=matcher.fit(glyph(3),[-.2,.1]);b=matcher.fit(glyph(3),[10,-20])
    assert a==b and a['digit']==3


@pytest.mark.parametrize('scale',[.6,1.,1.7])
def test_identity_validation_is_explicitly_not_metric_size(matcher,scale):
    mask=glyph(7,scale);fit=matcher.fit(mask)
    assert fit['digit']==7
    result=matcher.validate_identity(mask,fit)
    assert result['valid'] and result['scale_invariant']
    assert not result['metric_size_validated'] and not result['uses_object_geometry']


def test_empty_masks_do_not_validate(matcher):
    result=matcher.validate_identity(np.zeros((50,50),bool),{'digit':3,'yaw_deg':0})
    assert not result['valid']

@pytest.mark.parametrize('digit',[2,3,5,6])
def test_thick_printed_glyphs_at_unbanked_stroke_and_resolution(matcher,digit):
    from PIL import Image,ImageDraw,ImageFont
    path=Path(__file__).resolve().parents[1]/'asset_revision/physicalrsi_observed_geometry/fonts/DejaVuSans-Bold.ttf'
    font=ImageFont.truetype(str(path),57)
    image=Image.new('L',(100,100));draw=ImageDraw.Draw(image)
    draw.text((12,10),str(digit),font=font,fill=255,stroke_width=3,stroke_fill=255)
    rotated=cv2.warpAffine(np.asarray(image),cv2.getRotationMatrix2D((50,50),13,1),(100,100))
    assert matcher.fit(rotated>100)['digit']==digit
