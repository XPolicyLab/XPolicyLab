"""Remove rank/CAD height overrides while retaining the existing RGB estimator.

The contour estimator's camera/support migration and CAD shape registration are
separate outstanding work. This edit does not certify metadata-free nesting.
"""
import ast


def edit_nesting_heights(path,text):
    tree=ast.parse(text)
    if path.endswith('/perception.py'):
        count=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and ast.unparse(node.targets[0])=="p['calibrated_height_m']":
                node.value=ast.parse("p['height_m']",mode='eval').body;count+=1
        if count!=1:raise ValueError('Expected one nesting rank height override')
    elif path.endswith('/object_observer.py'):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ObjectObserver')
        qualify=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='qualify')
        qualify.body[:0]=ast.parse('''if not doll.get('geometric_valid', False):
    return None, {'valid': False, 'reason': 'observed_height_invalid_no_CAD_fallback'}''').body
        count=0
        for node in ast.walk(qualify):
            if isinstance(node,ast.keyword) and node.arg=='calibrated_height_m':
                node.value=ast.parse("float(doll['height_m'])",mode='eval').body;count+=1
        if count!=1:raise ValueError('Expected one nesting CAD height override')
        tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='profile_geometry')]
        observe=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
        class RemoveFallback(ast.NodeTransformer):
            def visit_If(self,node):
                condition=ast.unparse(node.test)
                if condition=="not doll['geometric_valid']":return None
                if 'height_only_invalid(doll)' in condition:
                    node.test=ast.parse("doll is None or not doll['geometric_valid']",mode='eval').body
                return self.generic_visit(node)
        RemoveFallback().visit(observe)
    elif path.endswith('/model.py'):
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpisodePolicy')
        for node in ast.walk(cls):
            if isinstance(node,ast.Assign) and len(node.targets)==1:
                target=ast.unparse(node.targets[0])
                if target=='self.height':
                    node.value=ast.parse("self.source['height_m']",mode='eval').body
                if target=='height' and ast.unparse(node.value)=='float(np.max(self.geometry.points[rank][:, 2]))':
                    node.value=ast.parse("float(d['height_m'])",mode='eval').body
    else:return text
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
