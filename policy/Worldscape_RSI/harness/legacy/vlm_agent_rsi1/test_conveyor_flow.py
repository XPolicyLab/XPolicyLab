from conveyor_flow import validate

def rejected(p, s):
    try: validate(p, s)
    except ValueError: return
    raise AssertionError('Expected rejection')

rejected({'stage':'transfer','commands':['wait']}, {'stage':'basket'})
rejected({'stage':'intercept','commands':['wait']}, {'stage':'basket'})
rejected({'stage':'transfer','commands':['left gripper open']}, {'stage':'transfer'})
rejected({'stage':'place','commands':['right gripper open']}, {'stage':'place'})
p,s=validate({'stage':'basket','macro':{'name':'grasp_probe','arm':'right','retreat_cm':[3,-4,3]}}, {})
assert len(p['commands']) == 2
p,s=validate({'stage':'intercept','evidence':{'basket_retained':'Basket moved with right wrist in two RGB observations.'},'commands':['wait 10']},s)
assert s['stage']=='intercept'
print('Conveyor stage/retention/release/macro tests PASS')
