# Курсовой проект по теме "Анализ эффективности архитектур визуального распознавания речи"

- дописать нормально импорт общего скрипта метрик
- дописать скрипт формирования таблички + построения графиков ?
- дописать скрипт для скачивания предобученной модели

- возможно добавить еще несколько архитектур моделей
## Клонирование репозитория
Для локальной работы склонируйте репозиторий с помощью команды:
```bash
git clone https://github.com/sadevans/test_lipreading.git
```

После этого перейдите в рабочую директорию:
```bash
cd test_lipreading
```

## Установка необходимых зависимостей
Необходимо создать виртуальную среду:

```bash
python -m venv venv
```

Далее активируйте свиртуальную среду:
```bash
source venv/bin/activate
```

Модель `auto-vsr` требует установки дополнительных пакетов. Скачать их придется с помощью клонирования соответствующих реплзиториев.
Для начала установите пакет `fairseq`. Для этого необходимо запустить ряд команд в своем терминале:
```bash
cd auto_vsr/
git clone https://github.com/pytorch/fairseq
cd fairseq/
pip install --editable ./
cd ..
```

Далее установите пакеты `face-recognition` и `face-alignment`.
Для этого поочереди склонируйте репозитории и установите все необходимые зависимости.
```bash
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
```
Потом:
```bash
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
```
Проверьте, что в папке `auto-vsr` появились `fairseq`, `face-alignment` и `face-recognition`.

После этого можно установить основные зависимости:
```bash
pip install -r requirements.txt
```

Не забудьте обновить пакет **hydra-core**
```bash
pip install hydra-core --upgrade
```
