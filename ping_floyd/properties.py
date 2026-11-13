"""Bounded property checks from explicit universally quantified contracts.

Only a whitelisted expression language is accepted. No property is guessed from
a function name or learned from public expected answers. Generated inputs are
ordered from small witnesses to larger cases to keep failure feedback useful.
"""
import ast
import re


BUILTINS = {'abs', 'len', 'sum', 'sorted', 'list', 'tuple', 'min', 'max', 'set', 'all', 'any'}
NODES = (ast.Expression, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
         ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
         ast.UnaryOp, ast.USub, ast.UAdd, ast.Not, ast.BoolOp, ast.And, ast.Or,
         ast.Call, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.keyword)


def safe_property(text, variable, symbols):
    tree = ast.parse(text, mode='eval')
    nodes = list(ast.walk(tree))
    if len(nodes)>100 or any(not isinstance(n,NODES) for n in nodes):
        raise ValueError('unsupported property expression')
    calls = set()
    for node in nodes:
        if isinstance(node,ast.Call):
            if (not isinstance(node.func,ast.Name) or node.func.id not in BUILTINS | symbols.keys()
                    or any(k.arg is None for k in node.keywords)):
                raise ValueError('property calls only named source functions and safe builtins')
            calls.add(node.func.id)
        if isinstance(node,ast.Name) and node.id not in {variable} | BUILTINS | symbols.keys():
            raise ValueError('unknown property variable')
        if isinstance(node,ast.Constant) and (type(node.value) not in {int,str,bool,type(None)}
                or isinstance(node.value,int) and abs(node.value)>10**12
                or isinstance(node.value,str) and len(node.value)>100):
            raise ValueError('property literal exceeds limits')
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            if not isinstance(node.right,ast.Constant) or type(node.right.value) is not int or not 0<=node.right.value<=4:
                raise ValueError('property power must be a small literal')
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):
            # Do not allow arbitrary sequence allocation in generated checks.
            if isinstance(node.left,(ast.List,ast.Tuple,ast.Constant)) and isinstance(node.right,ast.Constant):
                if type(node.right.value) is int and abs(node.right.value)>100:
                    raise ValueError('large property multiplication')
    used = calls & symbols.keys()
    if not used or any(len(symbols[n])!=1 for n in used):
        raise ValueError('property needs uniquely located source functions')
    if not isinstance(tree.body,(ast.Compare,ast.BoolOp)):
        raise ValueError('property must be a comparison')
    return used, ast.unparse(tree.body)


def quantified_checks(instruction, symbols):
    declarations = []
    # Python-style properties: explicitly universal, with a declared input domain.
    pattern = r'For (?:every|each) (integer|list of integers|nonempty list of integers) ([A-Za-z_]\w{0,20}),\s*`assert ([^`\n]{1,1200})`'
    for m in re.finditer(pattern,instruction,re.I):
        declarations.append((m[1].lower(),m[2],m[3],False))
    # Natural-language universal equations already used for contradiction checks.
    pattern = (r'For (?:every|each) integer ([A-Za-z]\w{0,15}),\s*([A-Za-z_]\w{0,39})'
               r'\(([^;,.\n]{1,80})\) must equal ([^;.\n]{1,80}?)(?: exactly)?\.')
    for m in re.finditer(pattern,instruction,re.I):
        declarations.append(('integer',m[1],f'{m[2]}({m[3]}) == ({m[4]})',True))
    bodies = []
    for domain,variable,expr,exact in declarations[:12]:
        try:
            used,expr = safe_property(expr,variable,symbols)
        except (SyntaxError,ValueError):
            continue
        # These are test inputs, never a table of answers.
        values = repr([0, 1, -1, 2, -2, 3, -3, 7, -7, 16, -16, 31, -31, 127, -127,
                       1000003, -1000003, 2**53-1, 2**53+1, -(2**53+1), 10**40+7, -(10**40+7)])
        if domain != 'integer':
            cases = [[],[0],[1],[-1],[0,0],[1,1],[-1,1],[2,0,-2],[3,1,3,0],list(range(11)),list(range(10,-1,-1))]
            if domain.startswith('nonempty'):
                cases = cases[1:]
            values = repr(cases)
        body = [f'from {symbols[n][0]} import {n}' for n in sorted(used)]
        if exact:
            # Natural-language equations use rational arithmetic. Explicit Python
            # assertions above retain Python's own semantics instead.
            class RationalDivision(ast.NodeTransformer):
                def visit_BinOp(self,node):
                    self.generic_visit(node)
                    if isinstance(node.op,ast.Div):
                        return ast.Call(func=ast.Name(id='_pf_fraction',ctx=ast.Load()),args=[node.left,node.right],keywords=[])
                    return node
            expr=ast.unparse(RationalDivision().visit(ast.parse(expr,mode='eval')))
            body.insert(0,'from fractions import Fraction as _pf_fraction')
        body.extend([f'for {variable} in {values}:',
                     f'    with self.subTest(witness={variable!s}):',
                     f'        self.assertTrue({expr}, {"required universal property: "+expr!r})'])
        bodies.append(body)
    return bodies
