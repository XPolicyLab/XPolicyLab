import unittest
from tasks.pour_by_language.skills.language.implementation import parse_instruction
EXPECTED = (('red', 'black'), ('turquoise', 'white'), ('violet', 'brown'))

class ObjectRoleTests(unittest.TestCase):
    def test_irrelevant_colours_do_not_shift_pairs(self):
        text = ('Ignore the red mark and the white stripe. '
                'Pour the red bottle into the black bowl, the turquoise bottle '
                'into the white bowl, and the violet bottle into the brown bowl.')
        self.assertEqual(parse_instruction(text), EXPECTED)

    def test_receiver_first_clauses_keep_instruction_order(self):
        text = ('Into the black bowl, pour the red bottle; '
                'into the white bowl, pour the turquoise bottle; '
                'into the brown bowl, pour the violet bottle.')
        self.assertEqual(parse_instruction(text), EXPECTED)

    def test_postnominal_colour_qualifiers(self):
        text = ('Pour the bottle that is red into the bowl coloured black; '
                'the bottle in colour turquoise into the bowl that is white; '
                'the bottle coloured violet into the bowl in colour brown.')
        self.assertEqual(parse_instruction(text), EXPECTED)

    def test_arrows_require_explicit_direction(self):
        self.assertEqual(parse_instruction('red -> black; turquoise → white; violet -> brown'), EXPECTED)
        with self.assertRaises(ValueError):
            parse_instruction('red black turquoise white violet brown')

    def test_wrong_roles_and_repeated_receivers_refuse(self):
        for text in [
            'White bottle into red bowl; turquoise bottle into black bowl; violet bottle into brown bowl.',
            'Red bottle into black bowl; turquoise bottle into black bowl; violet bottle into brown bowl.',
            'Red bottle, turquoise bottle, black bowl, white bowl, violet bottle, brown bowl.',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_instruction(text)

if __name__ == '__main__':
    unittest.main()
