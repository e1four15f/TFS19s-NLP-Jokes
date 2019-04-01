# TFS19s-NLP-Jokes
По мотивам <a href="https://amoudgl.github.io/blog/funnybot/">поста</a> запилить бота, который шутит шутейки.

### 1. Сбор данных (Alexandr Yu., Vasily Karmazin)
Сбор шутейк из <a href="https://vk.com">vk.com</a>, <a href="https://bash.im">bash.im</a>, <a href="https://2ch.hk">2ch.hk</a> и других подобных ресурсах

### 2. Препроцессинг данных (Alexandr Yu.)
 - Обработка полученных данны

 - Удаление мусора
 - ~~Нормализация текста~~ - если кто-то будет делать word based seq2seq, то он сделает нормализацию и токенизацию как он считает нужным


### 3. Создание архитектуры проекта (Alexandr Yu., Vasily Karmazin, Vlad Semak)
Изучение дополнительного материала
 - Статьи (<a href="https://guillaumegenthial.github.io/sequence-to-sequence.html">Seq2Seq with Attention and Beam Search</a>)
 - Похожие проекты 
 - Научные работы
 
### 4. Эксперимент 1. LSTM/RNN, Seq2Seq (Alexandr Yu., Vlad Semak)
Построение простой модели и оценка её качества

### 5. Эксперимент 2. Attention (Aexandr Yu., Vasily Karmazin)
Добавление attention-а к моделе 

### 6. Эксперимент 3. Language Model, Transformer (Alexandr Yu., Vasily Karmazin, Vlad Semak)
Изучение сложных архитектурных моделей, попытки использовать их в проекте 

### -1. (опционально) Телеграмм бот (Vasily Karmazin)
Создание телеграмм бота для удобства демонстрации проекта
