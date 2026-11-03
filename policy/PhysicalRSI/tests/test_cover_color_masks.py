from pathlib import Path
import importlib.util,sys
import cv2,numpy as np
REVISION=Path(__file__).resolve().parents[1]/'asset_revision'
sys.path.insert(0,str(REVISION))
spec=importlib.util.spec_from_file_location('cover_color_masks',REVISION/'overlays/cover_perception.py')
cover=importlib.util.module_from_spec(spec);spec.loader.exec_module(cover)


def test_weak_highlights_extend_seeded_cup_without_creating_objects():
    hsv=np.zeros((180,240,3),'uint8')
    # Three independently detected strong-color cup bodies with weak caps.
    for x in (25,95,165):
        hsv[70:105,x:x+25]=[30,150,220]
        hsv[55:70,x:x+25]=[30,60,245]
    # Same weak hue with no strong seed must remain excluded.
    hsv[125:155,50:80]=[30,60,245]
    rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    contours=cover.masks(rgb)['cup']
    assert len(contours)==3
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        assert y==55 and h==50 and w==25


def test_hysteresis_does_not_join_separated_bodies_through_other_hues():
    hsv=np.zeros((120,180,3),'uint8')
    hsv[30:65,20:45]=[30,150,220]
    hsv[30:65,100:125]=[30,150,220]
    hsv[40:50,45:100]=[10,120,200]
    assert len(cover.masks(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB))['cup'])==2


def test_red_shadow_face_is_recovered_from_bright_face_seed():
    hsv=np.zeros((120,180,3),'uint8')
    hsv[30:45,30:60]=[0,200,230]
    hsv[45:65,30:60]=[0,200,120]
    hsv[80:100,90:110]=[0,200,120]  # No bright seed: not another red block.
    contours=cover.masks(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB))['red']
    assert len(contours)==1
    assert cv2.boundingRect(contours[0])==(30,30,30,35)
