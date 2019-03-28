from src.parser import bash_parser as bp

p = bp.BashParser(list(range(0, 2920)))
p.parse(100)
