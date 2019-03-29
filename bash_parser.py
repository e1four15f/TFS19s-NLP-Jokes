from src.parser import bash_parser as bp


p = bp.BashParser(list(range(0, 2923)), db='jokes/jokes.db')
p.parse(10)
