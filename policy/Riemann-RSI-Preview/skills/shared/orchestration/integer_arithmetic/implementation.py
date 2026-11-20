import re


OPERATIONS = {'+': 'plus', 'plus': 'plus', '-': 'minus', 'minus': 'minus',
              '*': 'multiplication', '×': 'multiplication', 'times': 'multiplication',
              'multiplied by': 'multiplication', '/': 'division', '÷': 'division',
              'divided by': 'division'}
EXPRESSION = re.compile(r'(?<![\w.])(?P<left>\d{1,5})\s*'
    r'(?P<operation>multiplied\s+by|divided\s+by|plus|minus|times|[+*/×÷-])\s*'
    r'(?P<right>\d{1,5})\s*(?:=|equals|is\s+equal\s+to)\s*'
    r'(?P<result>-?\d{1,5})(?![\w.])', re.IGNORECASE)


def calculate(left_operand, operation, right_operand):
    if type(left_operand) is not int or type(right_operand) is not int:
        raise ValueError('The mature arithmetic procedure accepts integer operands only')
    if operation == 'plus':
        return left_operand + right_operand
    if operation == 'minus':
        return left_operand - right_operand if left_operand >= right_operand else None
    if operation == 'multiplication':
        return left_operand * right_operand
    if operation == 'division':
        return left_operand // right_operand if right_operand and left_operand % right_operand == 0 else None
    return None


def inconsistent_claim(instruction, request):
    if not isinstance(instruction, str) or not isinstance(request, dict):
        raise ValueError('An arithmetic check receives bounded public instruction and typed planner text')
    if not re.search(r'\b(?:equation|arithmetic|calculation|operator)\b', instruction, re.IGNORECASE):
        return False
    goal, reason = request.get('goal', ''), request.get('reason', '')
    if (not isinstance(goal, str) or len(goal) > 160
            or not isinstance(reason, str) or len(reason) > 320):
        raise ValueError('Arithmetic reasoning cannot expand the existing planner text allowance')
    for match in EXPRESSION.finditer(goal + '\n' + reason):
        operation = OPERATIONS[' '.join(match['operation'].lower().split())]
        expected = calculate(int(match['left']), operation, int(match['right']))
        if expected is None or expected != int(match['result']):
            return True
    return False
