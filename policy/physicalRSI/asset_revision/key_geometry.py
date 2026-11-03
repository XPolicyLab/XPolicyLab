"""Replace key/lock placement heights with current RGB metric observations.

This is an incomplete geometry migration: CAD shape/size consumers still remain.
"""
import ast


def edit_key_heights(path, source):
    if path.endswith('/insert_key/side_visible.py'):
        return 'from physicalrsi_observed_geometry.key_outline import held_outline\ndef held(rgb, relative, *, calibration, observed_outline):\n    return held_outline(rgb, relative, calibration, observed_outline)\n'
    if path.endswith('/insert_key/tooth_pose.py'):
        tree=ast.parse(source)
        tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='XZ' for t in n.targets))]
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='estimate')
        fn.args.kwonlyargs.append(ast.arg(arg='observed_outline'));fn.args.kw_defaults.append(None)
        def assigns(n,name):return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
        start=next(i for i,n in enumerate(fn.body) if assigns(n,'cam_in_key'))
        end=next(i for i,n in enumerate(fn.body) if assigns(n,'obj'))
        fn.body[start:end+1]=ast.parse('''obj=np.asarray(observed_outline,dtype=np.float64)
if obj.ndim!=2 or obj.shape[1]!=3 or len(obj)<8 or not np.isfinite(obj).all():
    return None,dict(reason='episode_outline_unresolved')
if np.linalg.matrix_rank(obj-obj.mean(0),tol=1e-8)<2:
    return None,dict(reason='episode_outline_degenerate')''').body
        text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
        text=text.replace('np.minimum([640, 480]', 'np.minimum([rgb.shape[1], rgb.shape[0]]')
        text=text.replace('RGB-to-public-CAD correspondences; calibrated wrist PnP.', 'Current RGB correspondences to the episode-observed planar key outline.')
        return text
    if not path.endswith('/insert_key/model.py'):return source
    tree=ast.parse(source)
    tree.body[1:1]=ast.parse('from physicalrsi_observed_geometry.calibrated_depth_scene import calibrated_depth_scene\nfrom physicalrsi_observed_geometry.key_outline import observe_outline, observed_key_tip').body
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef):continue
        if fn.name=='perceive':
            fn.args.kwonlyargs.append(ast.arg(arg='scene'));fn.args.kw_defaults.append(None)
            # Use the key and aperture's observed pixels, never a support-height constant.
            class RayHeight(ast.NodeTransformer):
                def visit_Call(self,node):
                    self.generic_visit(node)
                    if isinstance(node.func,ast.Name) and node.func.id=='ray':
                        if isinstance(node.args[1],ast.Constant) and node.args[1].value==.77:
                            node.args[1]=ast.parse('scene.point(bow)[2]',mode='eval').body
                        elif isinstance(node.args[1],ast.Constant) and node.args[1].value==.82:
                            node.args[1]=ast.parse('scene.point(ce[j])[2]',mode='eval').body
                    return node
            RayHeight().visit(fn)
            result=next(n for n in fn.body if isinstance(n,ast.Return))
            metadata=result.value.elts[2]
            assert isinstance(metadata,ast.Call) and isinstance(metadata.func,ast.Name) and metadata.func.id=='dict'
            metadata.keywords.append(ast.keyword(arg='observed_key_outline',value=ast.parse('observe_outline(camera, m, key).tolist()',mode='eval').body))
        elif fn.name=='wrist_center':
            fn.args.kwonlyargs.append(ast.arg(arg='plane_z'));fn.args.kw_defaults.append(None)
            class PlaneHeight(ast.NodeTransformer):
                def visit_Constant(self,node):
                    return ast.copy_location(ast.Name(id='plane_z',ctx=ast.Load()),node) if node.value==.77 else node
            PlaneHeight().visit(fn)
    class Calls(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='np.array'
                    and len(node.args)==1 and isinstance(node.args[0],ast.List)):
                try:values=ast.literal_eval(node.args[0])
                except (ValueError,TypeError):values=None
                if values==[0.,0.,-.05284674,1.]:
                    return ast.parse("np.r_[observed_key_tip(d['det']['observed_key_outline']), 1.]",mode='eval').body
            if isinstance(node.func,ast.Name) and node.func.id=='perceive':
                node.keywords.append(ast.keyword(arg='scene',value=ast.parse("calibrated_depth_scene(_frame_values['o'])",mode='eval').body))
            elif isinstance(node.func,ast.Name) and node.func.id in {'shaft_held','source_key_pose'}:
                node.keywords.append(ast.keyword(arg='observed_outline',value=ast.parse("d['det']['observed_key_outline']",mode='eval').body))
            elif isinstance(node.func,ast.Name) and node.func.id=='wrist_center':
                node.keywords.append(ast.keyword(arg='plane_z',value=ast.parse("d['key'][2, 3]",mode='eval').body))
            return node
    tree=Calls().visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
