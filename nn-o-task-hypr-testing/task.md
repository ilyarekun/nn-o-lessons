Тестирование гиперпараметров нейронных сетей с использованием PyTorch  
Срок сдачи: 26 марта 2025 года, 23:00  
Закрытие: 26 марта 2025 года, 23:00  

**Инструкции**  
В рамках этого задания вы будете тестировать гиперпараметры нейронных сетей (НС). Ваша цель — определить, какая конфигурация сети обеспечивает наилучшие результаты.

**Что должно быть включено в задание:**

- **Выбор набора данных:**  
  - Найдите набор данных, достаточно сложный (чтобы различия в точности были заметны при использовании разных сетей).  
  - Вы можете использовать общедоступные наборы данных (например, из UCI Machine Learning Repository, Kaggle и т.д.).  
  - Выполните EDA (статистические характеристики + визуализация с использованием pandas, matplotlib или seaborn).  

- **Тестирование различных топологий:**  
  - Создайте не менее 5 различных моделей с разными топологиями (например, количество слоев, количество нейронов в слое).  
  - Убедитесь, что остальные параметры (оптимизатор, скорость обучения, количество эпох, размер батча) остаются одинаковыми.  
  - Оцените точность каждой модели и выберите наиболее точную.  

- **Тестирование различных оптимизаторов:**  
  - Примените к лучшей топологии не менее 3 различных оптимизаторов.  
  - Сравните результаты и выберите оптимизатор, который обеспечивает наивысшую точность.  

- **Тестирование различных обучающих параметров:**  
  - С выбранной топологией и оптимизатором экспериментируйте с не менее чем 5 различными значениями скорости обучения (learning rate).  
  - Сравните, как изменяется производительность сети в зависимости от скорости обучения.  

- **Тестирование различных функций активации:**  
  - Для лучшей конфигурации (топология, оптимизатор, скорость обучения) протестируйте не менее 5 различных функций активации.  
  - Опишите в отчете функции активации, которые не упоминались ранее, даже если они не дали наилучших результатов.  
  - Оцените, какая комбинация обеспечивает наивысшую точность.  

**Реализация:**  
- Скрипт на Python (не Jupyter Notebook!) с реализацией вышеуказанных пунктов. (Название: Testovanie_Фамилия.py)  
- Если ваш набор данных требует этого, используйте early stopping или кросс-валидацию.  
- Не забудьте установить random seed.  
- Сохраните модель, которая показала наилучшую точность (для классификации) или наименьшую ошибку (для регрессии).  

**Отчет как часть сдачи:**  
- Подробное описание всех проведенных экспериментов с указанием использованных параметров, достигнутых результатов и визуализацией лучшей конфигурации (матрица ошибок может быть отображена с помощью matplotlib).  
- Обсуждение результатов — что повлияло на сети и почему вы выбрали конкретные настройки.  
- Используйте шаблон журнала IEEE в одностолбцовом формате. Сохраните PDF с названием: "Testovanie_Фамилия.pdf"