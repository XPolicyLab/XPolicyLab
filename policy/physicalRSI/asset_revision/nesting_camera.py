"""Bind nesting contour measurements to observed calibration and support."""
import ast


def edit_nesting_camera(path,text):
    tree=ast.parse(text)
    if path.endswith('/model.py'):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpisodePolicy')
        for name,code in [('reset','self.reader.head_size.support_plane = None\nself.reader.calibrated_observation = None'),
                          ('update_obs','self.reader.calibrated_observation = obs')]:
            method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
            method.body[:0]=ast.parse(code).body
    elif path.endswith(('/perception.py','/object_observer.py')):
        reader=path.endswith('/perception.py')
        tree.body[:0]=ast.parse('from physicalrsi_observed_geometry.upright_contour import refine_mask, pixel_scale').body
        for node in tree.body[:]:
            if isinstance(node,ast.ImportFrom) and node.module=='contours':
                node.names=[n for n in node.names if n.name!='refine_mask']
                if reader:tree.body.remove(node)
        class Bind(ast.NodeTransformer):
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id=='refine_mask':
                    node.keywords.extend([ast.keyword(arg='camera',value=ast.parse('self.head_size.camera',mode='eval').body),
                                          ast.keyword(arg='plane',value=ast.parse('self.head_size.support_plane',mode='eval').body)])
                return node
            def visit_Assign(self,node):
                if len(node.targets)!=1:return self.generic_visit(node)
                target=ast.unparse(node.targets[0])
                if target=='depth':
                    if reader:
                        expression=ast.unparse(node.value)
                        who=next(name for name in ('d','p','item') if f"{name}['bottom_pixel']" in expression)
                    else:who='doll'
                    node.value=ast.parse(f"pixel_scale(self.head_size.camera,self.head_size.support_plane,{who}['xy'])",mode='eval').body
                if reader and target=='local':return None
                if reader and target=='center' and 'local' in ast.unparse(node.value):
                    node.value=ast.parse('self.head_size.camera.project([[xy[0],xy[1],float(self.head_size.support_plane[:2] @ np.asarray(xy) + self.head_size.support_plane[2])+height*.5]])[0][0]',mode='eval').body
                return self.generic_visit(node)
            def visit_BinOp(self,node):
                if (isinstance(node.op,ast.Div) and isinstance(node.right,ast.Name)
                        and node.right.id in ('FX','focal')
                        and any(isinstance(n,ast.Name) and n.id=='depth' for n in ast.walk(node.left))):
                    return ast.copy_location(node.left,node)
                return self.generic_visit(node)
        tree=Bind().visit(tree)
        if reader:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Reader')
            read=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='read')
            read.body[:0]=ast.parse('''if getattr(self.head_size,'support_plane',None) is None:
    from physicalrsi_observed_geometry.support_observation import observe_support_geometry
    self.head_size.support_plane = observe_support_geometry(self.calibrated_observation,self.segment,occluders=('robot arm','nesting doll'))['plane']''').body
            remaining={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)} & {'CAM','ROT','FX','TABLE','ray'}
            if remaining:raise ValueError('Unmigrated nesting perception calibration: '+str(remaining))
    else:return text
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
