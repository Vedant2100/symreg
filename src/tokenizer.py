import sympy as sp
import re
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class EquationTokenizer:
    def __init__(self):
        self.vocab = {}
    
    def sympy_to_prefix(self, node):
        tokens = []
        if isinstance(node, sp.Function):
            tokens.append(node.func.__name__)
            for arg in node.args:
                tokens.extend(self.sympy_to_prefix(arg))
        elif isinstance(node, sp.Add):
            tokens.append('add')
            for arg in node.args:
                tokens.extend(self.sympy_to_prefix(arg))
        elif isinstance(node, sp.Mul):
            tokens.append('mul')
            for arg in node.args:
                tokens.extend(self.sympy_to_prefix(arg))
        elif isinstance(node, sp.Pow):
            tokens.append('pow')
            for arg in node.args:
                tokens.extend(self.sympy_to_prefix(arg))
        elif isinstance(node, sp.Symbol):
            tokens.append(node.name)
        elif isinstance(node, sp.Number):
            if node.is_Integer or node.is_Rational:
                tokens.append(str(node))
            else:
                tokens.append('<C>')
        else:
            name = node.func.__name__ if hasattr(node, 'func') else str(node.__class__.__name__)
            tokens.append(name)
            if hasattr(node, 'args'):
                for arg in node.args:
                    tokens.extend(self.sympy_to_prefix(arg))
        
        return tokens

    def binarize_expr(self, expr):
        if expr.is_Atom:
            return expr
        
        args = [self.binarize_expr(arg) for arg in expr.args]
        
        if isinstance(expr, (sp.Add, sp.Mul)) and len(args) > 2:
            return self._binarize_op(expr.func, args)
        
        if expr == sp.pi:
            return sp.Symbol('pi')
        if expr == sp.E:
            return sp.Symbol('E')
            
        return expr.func(*args)

    def _binarize_op(self, op, args):
        if len(args) == 2:
            return op(*args)
        return op(args[0], self._binarize_op(op, args[1:]))

    def tokenize_formula(self, formula_str):
        formula_str = formula_str.replace('^', '**')
        
        # Identify all potential variable names (words starting with letters)
        # and force them to be Symbols in local_dict to avoid collision with SymPy functions
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula_str)
        local_dict = {word: sp.Symbol(word) for word in words if word not in ['exp', 'sqrt', 'log', 'sin', 'cos', 'tan', 'pi', 'E']}

        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(formula_str, transformations=transformations, local_dict=local_dict)

        bin_expr = self.binarize_expr(expr)
        tokens = self.sympy_to_prefix(bin_expr)
        return tokens
