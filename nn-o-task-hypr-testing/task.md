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
-  

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






data section: Rossmann Store Sales from Kaggle

Dataset Description

You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.
Files

    train.csv - historical data including Sales
    test.csv - historical data excluding Sales
    sample_submission.csv - a sample submission file in the correct format
    store.csv - supplemental information about the stores

Data fields

Most of the fields are self-explanatory. The following are descriptions for those that aren't.

    Id - an Id that represents a (Store, Date) duple within the test set
    Store - a unique Id for each store
    Sales - the turnover for any given day (this is what you are predicting)
    Customers - the number of customers on a given day
    Open - an indicator for whether the store was open: 0 = closed, 1 = open
    StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
    SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
    StoreType - differentiates between 4 different store models: a, b, c, d
    Assortment - describes an assortment level: a = basic, b = extra, c = extended
    CompetitionDistance - distance in meters to the nearest competitor store
    CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
    Promo - indicates whether a store is running a promo on that day
    Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
    Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
    PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

добавь тут картинки 
nn-lessons/nn-o-task-hypr-testing/results/store_df_corr.png
nn-lessons/nn-o-task-hypr-testing/results/train_df_corr.png
nn-lessons/nn-o-task-hypr-testing/results/merged_df_corr.png
nn-lessons/nn-o-task-hypr-testing/results/Sales_distribution.png
nn-lessons/nn-o-task-hypr-testing/results/Sales_by_week.png
nn-lessons/nn-o-task-hypr-testing/results/Promo_on_Sales.png
nn-lessons/nn-o-task-hypr-testing/results/Day_on_Sales.png


добавь эти значения 


конец секции о данных



topology testing:

prototype 1:
(запиши сюда описание слоев в модели вход 31 input 1 output
class ModelRegv0(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(ModelRegv0, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim + 10),
            nn.ReLU(),
            nn.Linear(self.input_dim + 10, self.input_dim + 10),
            nn.ReLU(),
            nn.Linear(self.input_dim + 10, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim)
        )
        
    def forward(self,x):
        out = self.layers(x)
        return out
)


добавь картинку 
nn-lessons/nn-o-task-hypr-testing/results/metrics_v0.png


prototype 2:
class ModelRegv1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2 + 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.input_dim * 2 + 2, self.input_dim + 1),  # Исправлено
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim + 1),  # Исправлено
            nn.Linear(self.input_dim + 1, (self.input_dim + 1) // 2),
            nn.ReLU(),
            nn.Linear((self.input_dim + 1) // 2, self.output_dim)
        )
        
    def forward(self, x):
        out = self.layers(x)
        return out

        nn-lessons/nn-o-task-hypr-testing/results/metrics_v1.png


prototype 3:
class ModelRegv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2),  # 31 -> 62
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(self.input_dim * 2, self.input_dim),  # 62 -> 31
            nn.ReLU(),
            nn.BatchNorm1d(self.input_dim),
            nn.Dropout(0.3),
            nn.Linear(self.input_dim, self.input_dim // 2),  # 31 -> 15
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim)  # 15 -> 1
        )
        
    def forward(self, x):
        out = self.layers(x)
        return out

          nn-lessons/nn-o-task-hypr-testing/results/metrics_v2.png


class ModelRegv3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModelRegv3, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 3),  
            nn.ReLU(),
            nn.Linear(self.input_dim * 3, self.input_dim * 2),  
            nn.ReLU(),
            nn.Linear(self.input_dim * 2, self.input_dim + 32),  
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout перед уменьшением размерности
            nn.Linear(self.input_dim + 32, self.input_dim // 2 + 16),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2 + 16, self.input_dim // 4 + 8),
            nn.ReLU(),
            nn.Linear(self.input_dim // 4 + 8, self.output_dim)  # Выходной слой
        )

    def forward(self, x):
        return self.layers(x)

          nn-lessons/nn-o-task-hypr-testing/results/metrics_v3.png
