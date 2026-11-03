"""Use observed transfer heights consistently in planning and execution."""


def replace_language_transfer_heights(text):
    replacements=[
        ('        for tilt in [35, 50, 65]:',
         '        from physicalrsi_observed_geometry.language_transfer import observed_basket_top, observed_transfer_heights\n        basket_top = observed_basket_top(self, bb)\n        for tilt in [35, 50, 65]:',1),
        ('            plan = [{', '            carry_z, lower_z = observed_transfer_heights(contact, basket_top, tilt)\n            plan = [{',1),
        ("('lift', x, y, 1.02)","('lift', x, y, carry_z)",1),
        ("('transport', ex, ey, 1.02)","('transport', ex, ey, carry_z)",1),
        ("('lower', ex, ey, 0.98)","('lower', ex, ey, lower_z)",1),
        ('        lower_z = 0.98\n','',1),
        ("            lower_z = relay['lower_z']", "            lower_z = relay['lower_z']\n            carry_z = relay['carry_z']",1),
        ("'lower_z': lower_z, 'relay_site':", "'lower_z': lower_z, 'carry_z': carry_z, 'relay_site':",1),
        ("a.get('lower_z', 0.98)","a['lower_z']",1),
        ('[x, y, 1.02]',"[x, y, a['carry_z']]",1),
        ('[bx, by, 1.02]',"[bx, by, a['carry_z']]",4),
    ]
    for old,new,count in replacements:
        if text.count(old)!=count:raise ValueError(f'Unexpected transfer pattern: {old}')
        text=text.replace(old,new)
    if '1.02' in text:raise ValueError('Fixed carry height remains')
    compile(text,'observed language transfers','exec');return text


def replace_language_relay_transfer(text):
    old='lower_z = z + support.height(sx, sy) - support.height(x, y) + 0.015'
    if text.count(old)!=1:raise ValueError('Expected observed relay release')
    text=text.replace(old,old+'\n            carry_z = max(z, lower_z) + 0.10')
    for old,new in [("('lift', x, y, 1.02)","('lift', x, y, carry_z)"),
                    ("('transport', sx, sy, 1.02)","('transport', sx, sy, carry_z)"),
                    ("'lower_z': lower_z, 'uv':", "'lower_z': lower_z, 'carry_z': carry_z, 'uv':")]:
        if text.count(old)!=1:raise ValueError(f'Unexpected relay transfer: {old}')
        text=text.replace(old,new)
    if '1.02' in text:raise ValueError('Fixed relay carry height remains')
    compile(text,'observed relay transfer','exec');return text
