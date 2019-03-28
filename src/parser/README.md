Модуль для парсинга


```python
from src.parser import bash_parser as bp
pages_num = list(range(0, 2920))
p = bp.BashParser(pages_num)
p.parse(100)
```
