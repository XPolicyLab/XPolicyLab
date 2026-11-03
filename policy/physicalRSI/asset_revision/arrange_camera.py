"""Bind arrange perception to each instance's current official camera."""
import ast


def edit_arrange_camera(relative,text):
    if relative.endswith('/arrange_rgb/perception.py'):
        tree=ast.parse(text);lines=text.splitlines(keepends=True)
        changes=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'FX','FY','CAM','ROT'} for t in node.targets):
                changes.append((node.lineno-1,node.end_lineno,[]))
            if isinstance(node,ast.FunctionDef) and node.name=='rectify':
                changes.append((node.lineno-1,node.end_lineno,["def rectify(rgb,z=.78,*,camera):\n    return rectify_support(rgb,camera,[0.,0.,float(z)])\n"]))
        if len(changes)!=5:raise ValueError('Unexpected arrange camera constants/rectifier')
        for start,end,value in sorted(changes,reverse=True):lines[start:end]=value
        text=''.join(lines)
        # All remaining calls belong to Reader; the module entry requires an
        # explicit camera, so no stale process-global calibration can be used.
        text=text.replace('top=rectify(', 'top=self.rectify(').replace('flat=rectify(', 'flat=self.rectify(')
        text=text.replace('class Reader:\n', '''class Reader:
    def rectify(self,rgb,z=.78):
        camera=getattr(self,'camera',None)
        if camera is None:raise GeometryUnavailable('Current arrange camera required')
        return rectify(rgb,z,camera=camera)

''')
        text='from physicalrsi_observed_geometry import GeometryUnavailable\nfrom physicalrsi_observed_geometry.planar_rectification import rectify_support\n'+text
        text=text.replace('RGB silhouettes, static camera calibration and table-plane geometry only.', 'RGB silhouettes projected using the current observed camera.')
        text=text.replace('# Public fixed head-camera calibration and tabletop plane, no live matrices/depth.','# Object projection heights below remain pending replacement by observations.')
    elif relative.endswith('/arrange_rgb/semantic_recall.py'):
        text=text.replace('from .perception import rectify,normalize','from .perception import normalize')
        text=text.replace('rectify(', 'reader.rectify(')
    elif relative.endswith('/arrange_rgb/single_env.py'):
        tree=ast.parse(text);lines=text.splitlines(keepends=True);count=0
        for node in sorted(ast.walk(tree),key=lambda n:getattr(n,'lineno',0),reverse=True):
            if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='obs' for t in node.targets):continue
            camera='None' if isinstance(node.value,ast.Constant) and node.value.value is None else 'Camera.from_observation(self.obs)'
            line=lines[node.end_lineno-1];position=node.end_col_offset
            lines[node.end_lineno-1]=line[:position]+'; self.reader.camera = '+camera+line[position:]
            count+=1
        if count!=2:raise ValueError('Expected arrange observation/reset camera binding')
        text=''.join(lines)
    return text
