import sympy as sp
import re
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class EquationTokenizer:
    def __init__(self):
        self.vocab = {}
    
    def _binarize_list(self, op_name, args):
        if len(args) == 1:
            return self.sympy_to_prefix(args[0])
        elif len(args) == 2:
            return [op_name] + self.sympy_to_prefix(args[0]) + self.sympy_to_prefix(args[1])
        else:
            return [op_name] + self.sympy_to_prefix(args[0]) + self._binarize_list(op_name, args[1:])

    def sympy_to_prefix(self, node):
        if node == sp.pi:
            return ['pi']
        elif node == sp.E:
            return ['E']
        elif isinstance(node, sp.Function):
            tokens = [node.func.__name__]
            for arg in node.args:
                tokens.extend(self.sympy_to_prefix(arg))
            return tokens
        elif isinstance(node, sp.Add):
            return self._binarize_list('add', node.args)
        elif isinstance(node, sp.Mul):
            return self._binarize_list('mul', node.args)
        elif isinstance(node, sp.Pow):
            tokens = ['pow']
            for arg in node.args:
                tokens.extend(self.sympy_to_prefix(arg))
            return tokens
        elif isinstance(node, sp.Symbol):
            return [node.name]
        elif isinstance(node, sp.Number):
            if node.is_Integer or node.is_Rational:
                return [str(node)]
            else:
                return ['<C>']
        else:
            name = node.func.__name__ if hasattr(node, 'func') else str(node.__class__.__name__)
            tokens = [name]
            if hasattr(node, 'args'):
                for arg in node.args:
                    tokens.extend(self.sympy_to_prefix(arg))
            return tokens

    def tokenize_formula(self, formula_str):
        formula_str = formula_str.replace('^', '**')
        
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula_str)
        local_dict = {word: sp.Symbol(word) for word in words if word not in ['exp', 'sqrt', 'log', 'sin', 'cos', 'tan', 'pi', 'E']}

        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(formula_str, transformations=transformations, local_dict=local_dict)

        tokens = self.sympy_to_prefix(expr)
        return tokens
