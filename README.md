# Used car predictions + EDA 
Итак, передо мной была задача на основе готового датасета провести его анализ и построить предсказывающую модель. Так же нужно было разместить готовое решение на платформе **_Streamlint_**.

1. В качестве готового [датасета](https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/cars.csv) были выбраны данные о ценах подержанных автомобилей с различными характеристиками.

2. После выбора датасета шёл его анализ. (Увидеть его можно по сслылке выше)
3. Дальше мне предстояло построение модели. Так как целевая переменная была цена авто, то модель должна была выполнять задачу регресии. В качестве базовой решения я выбрал XGBoostRegressor. Затем с помощью GridSearchCv были найдены хорошие гиперпараметры, что позволило добиться хорошего качества предсказания (r2_score = 0.96)
4. В завершении был реализован веб-интерфейс на основе предсказывающей модели -  https://mainpy-qttxa514tdj.streamlit.app/

---
* main.py - Исполнительный файл для Streamlint
* model.py - Файл для построения предсказания для Streamlint
* data - Папка с файлом с картинкой и файлом с моделью
