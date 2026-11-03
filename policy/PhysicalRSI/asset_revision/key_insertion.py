"""Replace the asset-derived insertion seat coordinate with bounded probing."""
import ast


def edit_key_insertion(path,text):
    if not path.endswith('/insert_key/model.py'):return text
    tree=ast.parse(text)
    tree.body[1:1]=ast.parse('from physicalrsi_observed_geometry.key_insertion import observed_insertion_offset, observed_approach_offset, insertion_feedback\nfrom physicalrsi_observed_geometry import GeometryUnavailable').body
    class Edit(ast.NodeTransformer):
        offsets=0
        approaches=0
        readiness=0
        timeout=0
        def visit_AugAssign(self,node):
            if ast.unparse(node.target)=='key[2, 3]' and isinstance(node.value,ast.Constant) and node.value.value==.077:
                self.approaches+=1
                node.value=ast.parse("observed_approach_offset(d['det']['observed_key_outline'], key[:3,:3])",mode='eval').body
                return node
            if ast.unparse(node.target)=='key[2, 3]' and isinstance(node.value,ast.Constant) and node.value.value==.02325:
                self.offsets+=1
                if self.offsets==1:
                    return ast.parse('''
try:
    key[2,3] += observed_insertion_offset(d['det']['observed_key_outline'], key[:3,:3])
    d['insertion_probe'] = {}
    d['insertion_ready_count'] = 0
except GeometryUnavailable as error:
    d['insertion_geometry_error'] = str(error)
    carried = False
    key = mat(s[b + '_ee_pose']) @ d['receiver_key_local']
''').body
                return ast.parse("key = mat(s[b + '_ee_pose']) @ d['receiver_key_local']").body
            return self.generic_visit(node)
        def visit_If(self,node):
            if (ast.unparse(node.test)=="phase == 'insert'" and len(node.body)==1
                    and isinstance(node.body[0],ast.Assign)
                    and ast.unparse(node.body[0].targets[0])=="d['insertion_ready_count']"):
                self.readiness+=1
                node.body=ast.parse('''
try:
    d['insertion_feedback'] = insertion_feedback(
        d.setdefault('insertion_probe', {}), mat(receiver) @ d['receiver_key_local'],
        mat(target) @ d['receiver_key_local'], d['hole'], d['det']['observed_key_outline'])
except GeometryUnavailable as error:
    d['insertion_feedback'] = {'sustained_stop_after_insertion': False, 'must_hold': True, 'error': str(error)}
d['insertion_ready_count'] = d.get('insertion_ready_count', 0) + 1 if d['insertion_feedback']['sustained_stop_after_insertion'] else 0
''').body
            result=self.generic_visit(node)
            if ast.unparse(node.test)=="phase in ['orient', 'align'] and t > 65":
                self.timeout+=1
                return [result,*ast.parse('''
if d['phase'] == 'insert' and (t > 120 or d.get('insertion_feedback', {}).get('travel_bound_reached', False) or d.get('insertion_feedback', {}).get('must_hold', False)):
    d['insertion_stop_reason'] = 'No qualified progress stop before observed travel bound or timeout'
    advance('hold')
''').body]
            return result
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='tip_plane_clearance_m' for k in node.keywords):
                node.keywords.append(ast.keyword(arg='feedback',value=ast.parse("d.get('insertion_feedback')",mode='eval').body))
            return self.generic_visit(node)
    edit=Edit();tree=edit.visit(tree)
    if (edit.offsets,edit.approaches,edit.readiness,edit.timeout)!=(2,2,1,1):raise ValueError('Unexpected key insertion control structure')
    output=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    if '0.02325' in output:raise ValueError('Asset seat coordinate remains')
    return output
