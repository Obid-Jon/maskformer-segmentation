# MaskFormer Segmentation

Дообучение MaskFormer для бинарной семантической сегментации. 

## О проекте

Взял предобученную модель `facebook/maskformer-swin-base-coco` и дообучил её 
на своём датасете для задачи бинарной сегментации (фон / объект). 

Ноутбук писался в Google Colab, обучал на T4. Полный цикл: загрузка данных, 
аугментации, обучение с warmup и early stopping, оценка метрик, визуализация.

## Что внутри

- Класс датасета на PyTorch с аугментациями через albumentations
- Обучение с AdamW + cosine annealing + warmup
- Early stopping
- Метрики: IoU, F1, Precision, Recall, Dice, Accuracy, Confusion Matrix
- Визуализация предсказаний и графиков обучения
- Сохранение лучшей модели по val loss

## Структура

    maskformer-segmentation/
      notebooks/maskformer_segmentation.ipynb   — основной ноутбук
      requirements.txt
      README.md
      .gitignore

## Данные

Датасет скачивается автоматически из Google Drive при запуске ноутбука 
(ссылка внутри). Формат:

    dataset/
      train/
        image/   — изображения (.jpg/.png)
        mask/    — маски (.png), значения 0 и 1

Маски бинаризуются: 0 — фон, всё остальное — объект.

## Установка

Для запуска локально:

    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate       # Linux/Mac

    pip install --upgrade pip
    pip install -r requirements.txt

Дальше открыть ноутбук в Jupyter или VS Code.

## Обучение в Colab

Ноутбук рассчитан на Google Colab с GPU T4. В Colab всё ставится через 
первую ячейку (`!pip install ...`). Если запускаете локально — раскомментируйте 
`%%capture` и уберите `!` из команд.

## Параметры обучения

    batch_size      = 4
    num_epochs      = 8
    learning_rate   = 5e-5
    weight_decay    = 0.01
    optimizer       = AdamW
    scheduler       = LinearWarmup + CosineAnnealing
    image_size      = 512
    early_stopping  = patience 7

Модель сохраняется в `best_model_full.pth` по лучшему val loss.

## Метрики

Считаются на валидации после обучения:

- IoU по классам и средний
- F1 (по классам и средний)
- Precision, Recall
- Dice coefficient
- Accuracy
- Confusion matrix

Результаты сохраняются в `metrics_results.txt`.

## Результаты

Пример визуализации предсказаний и графиков обучения — в папке 
`docs/images/`.

(Если получилось что-то интересное по метрикам — впишите сюда цифры. 
Например: Mean IoU на валидации 0.ХХ, Mean Dice 0.ХХ.)

## Известные проблемы

- Датасет качается с Google Drive через gdown. Если ссылка перестанет 
  работать — пишите issue.
- Обучение рассчитано на GPU. На CPU будет очень медленно.
- В `SimpleSegmentationDataset` обработка выхода процессора сделана 
  через `squeeze` — работает для batch_size=1, при других значениях 
  может потребоваться правка collate_fn.

## Что можно улучшить

- Вынести код из ноутбука в модули (`src/dataset.py`, `src/train.py`)
- Добавить логирование через `logging` вместо print
- Написать unit-тесты для датасета и метрик
- Пробовать другие backbones (Swin-L, BEiT)
- Добавить mixed precision (AMP) для ускорения

## Лицензия

MIT.

## Автор

Obid-Jon
https://github.com/Obid-Jon