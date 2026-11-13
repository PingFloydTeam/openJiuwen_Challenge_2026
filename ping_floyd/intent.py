"""Conservative semantic guards for explicit symbolic questions.

This is not a natural-language theorem prover. Unknown formulations are left to
the agent; recognized objectives cannot be replaced by unrelated calculations.
"""
import ast
import re

from .constraints import affine
from .finite import expression


def equation_key(text):
    left,sep,right = text.replace('==','=').partition('=')
    if not sep:
        return None
    try:
        terms,constant = affine(f'({left})-({right})')
    except (SyntaxError, ValueError):
        return None
    values = [*sorted(terms.items()), ('',constant)]
    scale = next((v for _,v in values if v),1)
    return tuple((k,str(v/scale)) for k,v in values if v)


def explicit_equations(question):
    atom = r'(?:[A-Za-z]\w{0,15}|\d+|\([^;,\n=<>!?]{1,80}\))'
    expr = r'[+-]?\s*'+atom+r'(?:\s*(?:\*\*|//|[+*/%\-])\s*[+-]?\s*'+atom+r')*'
    found = []
    for match in re.finditer(rf'(?<![\w.])({expr})\s*(?<![<>!=])=(?!=)\s*({expr})(?!\w|\.\d)',question):
        text = match[1].strip()+'='+match[2].strip()
        key = equation_key(text)
        if key is not None and any(k for k,_ in key):
            found.append((text,key))
    return found


def symbolic_goal(question):
    if re.match(r'\s*(?:how many|count)\b',question,re.I) and re.search(r'\b(?:integer|integers)\b',question,re.I) and '=' in question:
        return '', 'count'
    summed = re.search(r'\bsum\s+(.{1,150}?)\s+over\s+(?:all|the)\s+(?:feasible|valid|admissible)\b',question,re.I)
    if summed:
        try:
            target=summed[1].strip(' `')
            names={n.id for n in ast.walk(ast.parse(target,mode='eval')) if isinstance(n,ast.Name)}
            if names and '=' in question[:summed.start()] and all(re.search(r'\b'+re.escape(n)+r'\b',question[:summed.start()]) for n in names):
                expression(target,names)
                return target,'sum'
        except (SyntaxError,ValueError):
            pass
    # Only the last explicit request, followed by a complete arithmetic expression.
    cues = list(re.finditer(r'\b(?:return|find|determine|compute|what is|give)\s+', question, re.I))
    for cue in reversed(cues):
        tail = question[cue.end():].strip().rstrip('.?')
        tail = re.split(r'\s+as (?:an? |the )?(?:exact |reduced )?(?:fraction|integer|number)\b', tail, flags=re.I)[0]
        tail = re.sub(r'^(?:the\s+)?', '', tail, flags=re.I)
        mode = 'unique'
        prefix = re.match(r'(maximum|minimum|smallest|largest)(?:\s+(?:achievable|possible|value of))?\s+', tail, re.I)
        if prefix:
            mode = 'max' if prefix[1].lower() in {'maximum','largest'} else 'min'
            tail = tail[prefix.end():]
        tail = re.sub(r'^(?:unique\s+)?value\s+of\s+', '', tail, flags=re.I)
        tail = tail.strip(' `')
        try:
            tree = ast.parse(tail, mode='eval')
            names = {n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
            if not names or len(names)>6:
                continue
            prefix_text = question[:cue.start()]
            if (not re.search(r'[A-Za-z]\w*\s*(?:[+*/%\-][^.;?]{0,50})?(?:==|=|<=|>=|<|>)', prefix_text)
                    or any(not re.search(r'\b'+re.escape(n)+r'\b',prefix_text) for n in names)):
                continue
            expression(tail, names)
        except (SyntaxError, ValueError):
            continue
        return tail, mode
    return None


def equivalent(left, right):
    try:
        if ast.dump(ast.parse(left,mode='eval')) == ast.dump(ast.parse(right,mode='eval')):
            return True
        return affine(left) == affine(right)
    except (SyntaxError, ValueError):
        return False


def check_goal(goal, tool, args, required=(), question=''):
    """Reject a known objective mismatch before consuming a tool result."""
    if goal is None:
        return
    target, mode = goal
    if tool in {'read','discover','schema'}:
        return
    if tool == 'finite_search':
        if args.get('mode','unique') != mode or mode != 'count' and not equivalent(args.get('target',''),target):
            raise ValueError(f'goal is {mode} of {target}; finite_search target/mode must match it')
    elif tool == 'linear' and mode == 'unique' and equivalent(args.get('variable',''),target):
        pass
    else:
        raise ValueError(f'goal is {mode} of {target}; use constraints and that target, not an unrelated scalar calculation')
    supplied = {equation_key(text) for text in args.get('constraints',args.get('equations',[]))}
    missing = [text for text,key in required if key not in supplied]
    if missing:
        raise ValueError('include the explicit task equations: '+ '; '.join(missing))
    if tool == 'finite_search':
        declaration = re.search(r'([A-Za-z]\w*(?:\s*,\s*[A-Za-z]\w*){1,5})\s+(?:are\s+integers\s+|lie\s+)?in\s*\[(-?\d+)\s*,\s*(-?\d+)\]',question)
        if declaration:
            bounds=[int(declaration[2]),int(declaration[3])]
            if any(args.get('domains',{}).get(n.strip()) != bounds for n in declaration[1].split(',')):
                raise ValueError('use the task\'s complete declared domains; narrowing them can fabricate uniqueness')
        predicates=[]
        for match in re.finditer(r'\b([A-Za-z]\w*)\s+mod\s+(\d+)\s*=\s*(-?\d+)',question):
            predicates.append(f'{match[1]}%{match[2]}=={match[3]}')
        for match in re.finditer(r'\b([A-Za-z]\w*)\s+is\s+(odd|even)\b',question,re.I):
            predicates.append(f'{match[1]}%2=={int(match[2].lower()=="odd")}')
        keys=set()
        for text in args.get('constraints',[]):
            try:
                tree=ast.parse(re.sub(r'(?<![<>=!])=(?!=)','==',text),mode='eval').body
            except SyntaxError:
                continue
            nodes=tree.values if isinstance(tree,ast.BoolOp) and isinstance(tree.op,ast.And) else [tree]
            keys.update(ast.dump(n) for n in nodes)
        missing=[p for p in predicates if ast.dump(ast.parse(p,mode='eval').body) not in keys]
        if missing:
            raise ValueError('include stated modular/parity conditions: '+ '; '.join(missing))


def bounded_inverse_evidence(question,args,value,residue):
    """An exact finite inverse can satisfy the same parity obligation as a macro."""
    target=args.get('target','')
    if args.get('mode','unique')!='unique' or target not in args.get('domains',{}):
        return False
    interval=re.search(r'\[(-?\d+)\s*,\s*(-?\d+)\]|\bbetween (-?\d+) and (-?\d+)\b',question,re.I)
    if interval is None:
        return False
    bounds=[int(v) for v in interval.groups() if v is not None]
    if args['domains'][target]!=bounds or not re.fullmatch(r'-?\d+',str(value)) or int(value)%2!=residue:
        return False
    wanted=ast.dump(ast.parse(f'{target}%2=={residue}',mode='eval').body)
    for text in args.get('constraints',[]):
        try:
            tree=ast.parse(re.sub(r'(?<![<>=!])=(?!=)','==',text),mode='eval').body
        except SyntaxError:
            continue
        nodes=tree.values if isinstance(tree,ast.BoolOp) and isinstance(tree.op,ast.And) else [tree]
        if wanted in {ast.dump(n) for n in nodes}:
            return True
    return False
