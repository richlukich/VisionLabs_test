# VisionLabs_test

Файлы:

-   **cls_labeled** - папка содержащая размеченный датасет и скрипт для обучения классификатора (train.py)
-   **vae** - папка содержащая скрипт обучения VAE (train_vae.py), классификатора (train_cls.py), а также частично размеченный датасет
-   **blip2_image_text_matching.ipynb** - скрипт для полуавтоматической разметки датасета с помощью BLIP-2, см. инструкцию к установке в [LAVIS](https://github.com/salesforce/LAVIS?tab=readme-ov-file#installation)
-   **inference.py** - финальный скрипт, содержащий класс OpenEyesClassificator
-   **VisionLabs_Report.pdf** - отчет по заданию

Веса модели находятся по [ссылке](https://drive.google.com/file/d/10m4rM71dc3r7WwoznxngTAOyik3T5OyA/view?usp=sharing)

## Установка
-   Создание окружения:
    ```bash
    conda create --name vision_labs python=3.9 -y
    conda activate vision_labs
    ```
-   Установка Pytorch:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

## Запуск

```bash
python inference.py
```
