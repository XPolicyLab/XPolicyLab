"""Preserve relative control gates while replacing fixed support elevation."""
import ast


def edit_nesting_support(path,text):
    tree=ast.parse(text)
    if path.endswith('/model.py'):
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpisodePolicy')
        read=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='read')
        assignment=next(n for n in read.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='p')
        read.body[read.body.index(assignment)]=ast.Try(body=[assignment],handlers=[ast.ExceptHandler(
            type=ast.Name(id='GeometryUnavailable',ctx=ast.Load()),name='error',
            body=ast.parse("p={'valid':False,'dolls':[],'visible_dolls':[],'failure':str(error),'geometry_unavailable':True}").body)],orelse=[],finalbody=[])
        read.body.insert(0,ast.parse('from physicalrsi_observed_geometry import GeometryUnavailable').body[0])
        for method in owner.body:
            if not isinstance(method,ast.FunctionDef):continue
            has_table=any(isinstance(n,ast.Name) and n.id=='TABLE' for n in ast.walk(method))
            if has_table and method.name not in {'begin_observation_park','begin_pick','empty_clearance','advance'}:
                raise ValueError('Unmapped nesting support use: '+method.name)
            xy="d['xy']" if method.name=='begin_observation_park' else 'self.xy'
            class Bind(ast.NodeTransformer):
                def visit_Name(self,node):
                    if node.id=='TABLE':
                        return ast.parse(f'support_height(self.reader.head_size.support_plane,{xy})',mode='eval').body
                    return node
                def visit_Assign(self,node):
                    if len(node.targets)==1 and ast.unparse(node.targets[0])=='floor':
                        node.value=ast.parse("max(support_height(self.reader.head_size.support_plane,d['xy'])+float(d['height_m']) for d in self.ordered)+.035",mode='eval').body
                    if len(node.targets)==1 and ast.unparse(node.targets[0])=='self.dest':
                        old=ast.unparse(node.value)
                        if old=='self.grasp + np.r_[self.drop - self.xy, 0.01]':
                            node.value=ast.parse('self.grasp + np.r_[self.drop-self.xy,.01+support_height(self.reader.head_size.support_plane,self.drop)-support_height(self.reader.head_size.support_plane,self.xy)]',mode='eval').body
                    return self.generic_visit(node)
            Bind().visit(method)
    elif path.endswith('/wrist_geometry.py'):
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WristGeometry')
        lift=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='fit_lift')
        offsets={.765:0.,.756:-.009,1.004:.239,.755:-.01,1.005:.24,.845:.08}
        class Elevation(ast.NodeTransformer):
            def visit_Constant(self,node):
                if isinstance(node.value,float) and node.value in offsets:
                    return ast.parse('support_z'+(f'+({offsets[node.value]!r})' if offsets[node.value] else ''),mode='eval').body
                return node
        Elevation().visit(lift)
        lift.body[:0]=ast.parse('support_z=support_height(self.support_plane,xy)').body
        carried=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='fit_carried')
        class Carried(ast.NodeTransformer):
            def visit_Constant(self,node):
                if isinstance(node.value,float) and node.value==.78:
                    return ast.parse('support_height(self.support_plane,fit.x[:2])+.015',mode='eval').body
                return node
        Carried().visit(carried)
    elif path.endswith('/deploy.py'):
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict) and len(node.keys)==1 and isinstance(node.keys[0],ast.Constant) and node.keys[0].value=='color':
                for key in ('intrinsic_matrix','extrinsic_matrix'):
                    node.keys.append(ast.Constant(key))
                    node.values.append(ast.parse(f"raw['vision'][k][{key!r}]",mode='eval').body)
        return ast.unparse(ast.fix_missing_locations(tree))+'\n'
    elif not path.endswith('/object_observer.py'):
        return text
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in {'contours','attachment'})]
    tree.body.insert(0,ast.parse('from physicalrsi_observed_geometry.upright_contour import support_height').body[0])
    if any(isinstance(n,ast.Name) and n.id=='TABLE' for n in ast.walk(tree)):
        raise ValueError('Fixed nesting support reference remains')
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
