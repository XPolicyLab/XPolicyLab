"""Replace stored nesting profiles with episode-local observed surface samples."""
import ast


def edit_nesting_shapes(path,text):
    tree=ast.parse(text)
    def cls(name):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    def method(owner,name):return next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
    if path.endswith('/wrist_geometry.py'):
        owner=cls('WristGeometry')
        method(owner,'__init__').body=ast.parse('self.points = []\nself.profile_evidence = []\nself.support_plane = None').body
        hull=method(owner,'hull')
        for node in ast.walk(hull):
            if isinstance(node,ast.IfExp) and ast.unparse(node.test)=='len(xy) > 2':
                node.orelse=ast.parse('float(self.support_plane[:2] @ np.asarray(xy[:2]) + self.support_plane[2])',mode='eval').body
    elif path.endswith('/model.py'):
        owner=cls('EpisodePolicy')
        for node in ast.walk(owner):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and ast.unparse(node.targets[0])=='self.geometry':
                node.value=ast.parse('self.reader.head_size.geometry',mode='eval').body
        method(owner,'reset').body[:0]=ast.parse('self.geometry.points = []\nself.geometry.profile_evidence = []\nself.geometry.support_plane = None').body
        # Crop height and clearance derive from observed extents, not partial point samples.
        class Heights(ast.NodeTransformer):
            def visit_Call(self,node):
                expression=ast.unparse(node)
                if expression=='float(np.max(self.geometry.points[rank][:, 2]))':
                    return ast.parse("float(self.ordered[rank]['height_m'])",mode='eval').body
                if expression=='max((float(np.max(points[:, 2])) for points in self.geometry.points))':
                    return ast.parse("max(float(d['height_m']) for d in self.ordered)",mode='eval').body
                return self.generic_visit(node)
        tree=Heights().visit(tree)
    elif path.endswith('/perception.py'):
        owner=cls('Reader');read=method(owner,'read')
        if not isinstance(read.body[-1],ast.Return):raise ValueError('Expected final nesting perception return')
        read.body[-1:-1]=ast.parse('''if decoded['valid'] and not self.head_size.geometry.points:
    from physicalrsi_observed_geometry.object_points import observed_object_points
    points,evidence = observed_object_points(self.calibrated_observation,decoded['dolls'],self.head_size.support_plane)
    self.head_size.geometry.points = points
    self.head_size.geometry.profile_evidence = evidence
    self.head_size.geometry.support_plane = self.head_size.support_plane
    decoded['shape_evidence'] = evidence''').body
        decode=method(owner,'decode')
        for node in ast.walk(decode):
            if isinstance(node,ast.If) and ast.unparse(node.test)=='fitted is not None':
                for assignment in node.orelse:
                    if isinstance(assignment,ast.Assign) and ast.unparse(assignment.targets[0])=='valid':
                        assignment.value=ast.Constant(False)
    elif path.endswith('/head_size.py'):
        owner=cls('HeadSize')
        method(owner,'assign').body[:0]=ast.parse('''if not self.geometry.points:
    from physicalrsi_observed_geometry.object_points import initial_size_order
    return initial_size_order(dolls)''').body
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name=='descriptor':
                node.body=ast.parse('''base_z=float(self.support_plane[:2] @ np.asarray(xy) + self.support_plane[2])
uv,_=self.camera.project(g.points[rank]+[xy[0],xy[1],base_z])
return g.descriptor(cv2.convexHull(uv.astype(np.float32)))''').body
    elif path.endswith('/profile_subset.py'):
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_profiles')
        function.body[:0]=ast.parse('''if not head.geometry.points:
    from physicalrsi_observed_geometry.object_points import initial_size_order
    return initial_size_order(dolls)''').body
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name=='descriptor':
                node.body=ast.parse('''base_z=float(head.support_plane[:2] @ np.asarray(xy) + head.support_plane[2])
uv,_=head.camera.project(g.points[rank]+[xy[0],xy[1],base_z])
return g.descriptor(cv2.convexHull(uv.astype(np.float32)))''').body
            if isinstance(node,ast.keyword) and node.arg=='calibrated_height_m':
                node.value=ast.parse("dolls[i]['height_m']",mode='eval').body
    elif path.endswith('/object_observer.py'):
        hull=method(cls('ObjectObserver'),'hull')
        class Support(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='TABLE':return ast.parse('float(self.head_size.support_plane[:2] @ np.asarray(xy) + self.head_size.support_plane[2])',mode='eval').body
                return node
        Support().visit(hull)
    else:return text
    if path.endswith(('/head_size.py','/profile_subset.py')):
        tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='contours')]
    if isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):
        tree.body[0].value.value='Within-episode RGB surface samples; no stored object profile. Partial observed surfaces only.'
    text=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    return (text.replace('public profile assignment','within-episode observed profile assignment')
            .replace('DINO_RGB_GrabCut_public_profiles','DINO_RGB_GrabCut_observed_profiles')
            .replace('qualified static profile ambiguity resolution','qualified episode profile ambiguity resolution'))
