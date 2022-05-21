# SkillFactory_dev
SkillFactory

## DS_Project
В текущем репозитории содержится исходный код дипломной работы студентки Брусовой Галины
Папка - Final_Project

### Описание задачи
Прогнозирование стоимости квадратного метра жилья на данных Яндекс Недвижимости за 2017-2019 года.
Данная задача была представлена на [хакатоне](https://yandex.ru/promo/realty/hacktherealty).
Выбранная метрика - RMSE.
Финальная архитектура модели - градиентный бустинг на решающих деревьях.

Качество на тестовой выборке (2019 годе):
R^2 :  0.993
MAE : 3449.632
RMSE: 8410.706
MAPE: 1.625

### Описание файлов
* <code>[Добавление фичей + EDA](/Final_Project/EDA.ipynb)</code>
* <code>[Обучение и тюнинг модели](/Final_Project/Model_Pipeline.ipynb)</code>
* <code>[Датасет для обучения модели](/Final_Project/data_sf/df_price.csv)</code>
* consultant2017-2019.json - файлы с данными о празниках и выходных днях Consultant

### Ссылки:
* <code>[Отчет pandas-profiling](https://drive.google.com/file/d/1eBmG29eZfUCSaaRqIEI5vlMlEpyaxHuh/view?usp=drivesdk)</code>
* <code>[Отчет Yandex DATA Lens](https://datalens.yandex.ru/ud7ndrq1r6zsm-linasf)</code>
* <code>[Google Colab](https://drive.google.com/drive/folders/14nrHxeaabvXwKkEBMbrxH1qVsJxg9Iqo)</code>  
  
  
### TODO
Добавится презентация
